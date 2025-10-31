import argparse
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import evaluate
from peft import get_peft_model, LoraConfig, TaskType, IA3Config

def device_choice():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class SoftPrompt(nn.Module):
    def __init__(self, prompt_length: int, hidden_size: int, device=None):
        super().__init__()
        self.prompt_length = prompt_length
        self.embedding = nn.Embedding(prompt_length, hidden_size)
        if device is not None:
            self.to(device)
    def forward(self, batch_size: int, device=None):
        dev = device if device is not None else self.embedding.weight.device
        idx = torch.arange(self.prompt_length, device=dev).unsqueeze(0).expand(batch_size, -1)
        return self.embedding(idx)  # [batch, prompt_length, hidden]

def prepare_data(dataset_name, max_length=64):
    dataset = load_dataset("glue", dataset_name)
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    return tokenized["train"], tokenized["validation"]

def build_model(peft, tuning, device, args, prompt_length=10):
    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

    # apply PEFT wrapping early
    if peft == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        model = get_peft_model(model, lora_config)
    elif peft == "ia3":
        ia3_config = IA3Config(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            target_modules=["query", "value"]
        )
        model = get_peft_model(model, ia3_config)

    model.to(device)

    # Tuning mode parameter registration / requires_grad toggles
    if tuning == "full":
        for _, p in model.named_parameters():
            p.requires_grad = True
    elif tuning == "last":
        for name, p in model.named_parameters():
            p.requires_grad = ("classifier" in name)
    elif tuning == "soft":
        # register SoftPrompt as embedding submodule on embeddings so hooks see it
        hidden_size = model.config.hidden_size
        soft = SoftPrompt(prompt_length, hidden_size, device=device)
        # attach module
        model.bert.embeddings.add_module("soft_prompt", soft)
        # freeze everything except soft prompt and any LOra params
        for name, p in model.named_parameters():
            if ("lora" in name.lower()) or ("soft_prompt" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False
        # monkeypatch forward to prepend prompt embeddings (robust to inputs_embeds)
        original_forward = model.forward
        def new_forward(*args, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
            # infer input_ids from positional args if passed positionally
            if len(args) > 0 and input_ids is None and inputs_embeds is None:
                possible = args[0]
                if isinstance(possible, torch.Tensor):
                    input_ids = possible
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("neither input_ids nor inputs_embeds provided")
                inputs_embeds = model.bert.embeddings(input_ids=input_ids)
                dev = input_ids.device
            else:
                dev = inputs_embeds.device
            batch_size = inputs_embeds.size(0)
            prompt_emb = model.bert.embeddings.soft_prompt(batch_size, device=dev)
            inputs_embeds = torch.cat([prompt_emb, inputs_embeds], dim=1)
            # extend attention_mask/token_type_ids
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=dev)
            else:
                prefix_mask = torch.ones((batch_size, prompt_length), dtype=attention_mask.dtype, device=dev)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            if token_type_ids is None:
                token_type_ids = torch.zeros((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=dev)
            else:
                prefix_tt = torch.zeros((batch_size, prompt_length), dtype=token_type_ids.dtype, device=dev)
                token_type_ids = torch.cat([prefix_tt, token_type_ids], dim=1)
            dummy_input_ids = None
            if input_ids is None:
                dummy_input_ids = torch.zeros((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=dev)
            return original_forward(input_ids=dummy_input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, **kwargs)
        model.forward = new_forward
    elif tuning == "prefix":
        prefix_length_local = prompt_length
        hidden_size = model.bert.embeddings.word_embeddings.embedding_dim
        # create per-layer Embedding modules properly on device
        for idx, layer in enumerate(model.bert.encoder.layer):
            emb = nn.Embedding(prefix_length_local, hidden_size)
            nn.init.normal_(emb.weight, mean=0.0, std=hidden_size ** -0.5)
            emb = emb.to(device)
            # register module on layer
            layer.add_module("prefix_emb", emb)
        # freeze everything except LoRA and prefix embeddings
        for name, p in model.named_parameters():
            if ("lora" in name.lower()) or ("prefix_emb" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False
        # monkeypatch forward: insert per-layer prefix embeddings
        original_forward = model.forward
        def new_forward(input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
            unused_kwargs = kwargs
            if input_ids is None:
                raise ValueError("prefix-mode requires input_ids")
            batch_size = input_ids.size(0)
            dev = input_ids.device
            inputs_embeds = model.bert.embeddings(input_ids=input_ids)
            if attention_mask is None:
                base_attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=dev)
            else:
                base_attention_mask = attention_mask.to(dtype=torch.long, device=dev)
            if token_type_ids is None:
                token_type_ids = torch.zeros((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=dev)
            else:
                token_type_ids = token_type_ids.to(dtype=torch.long, device=dev)
            hidden_states = inputs_embeds
            for i, layer in enumerate(model.bert.encoder.layer):
                idx = torch.arange(prefix_length_local, device=dev).unsqueeze(0).expand(batch_size, -1)
                prefix_emb = layer.prefix_emb(idx)  # per-sample lookup
                hidden_states = torch.cat([prefix_emb, hidden_states], dim=1)
                prefix_mask = torch.ones((batch_size, prefix_length_local), dtype=base_attention_mask.dtype, device=dev)
                base_attention_mask = torch.cat([prefix_mask, base_attention_mask], dim=1)
                extended_attention_mask = model.bert.get_extended_attention_mask(base_attention_mask, (batch_size, hidden_states.size(1)), dev)
                hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
            pooled_output = model.bert.pooler(hidden_states)
            logits = model.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
            from types import SimpleNamespace
            if loss is None:
                return SimpleNamespace(logits=logits)
            else:
                return SimpleNamespace(loss=loss, logits=logits)
        model.forward = new_forward
    else:
        raise ValueError(f"unknown tuning mode: {tuning}")
    
    # Ensure backbone params are frozen by default; enable only adapter/prompt params
    if tuning != "full":
        for n, p in model.named_parameters():
            # keep LoRA/IA3 adapter params and any prompt/prefix params trainable, freeze the rest
            lower = n.lower()
            if ("lora" in lower) or ("ia3" in lower) or ("soft_prompt" in n) or ("prefix_emb" in n) or (tuning == "last" and "classifier" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

    # rebuild optimizer from trainable params
    trainable_params = [p for _, p in model.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found after applying tuning/peft configuration")
    
    # print number of trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)

    print(f"Total params: {total_params}, Trainable params: {trainable_count} ({100.0 * trainable_count / total_params:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    last_loss = None
    for batch in train_loader:
        # move tensors to device and pop labels
        labels = batch.pop("label")
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return last_loss

def evaluate_model(model, eval_loader, device):
    metric = evaluate.load("accuracy")
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            labels = batch.pop("label")
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            else:
                labels = labels.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=preds.cpu(), references=labels.cpu())
    return metric.compute()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", help="dataset name (options: sst2, qnli)")
    parser.add_argument("--tuning", type=str, default="soft", choices=["full","soft","prefix","last"])
    parser.add_argument("--peft", type=str, default=None, choices=[None,"lora","ia3"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--promptlength", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = device_choice()
    train_ds, eval_ds = prepare_data(args.dataset)
    batch_size = 1024
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size)

    model, optimizer, criterion = build_model(args.peft, args.tuning, device, args, prompt_length=args.promptlength)
    for i in range(args.epochs):
        loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        acc = evaluate_model(model, eval_loader, device)
        print(f"Epoch {i+1}/{args.epochs} — loss (one batch)={loss} -- eval_acc_sample={acc}")
    print(f"Finished: tuning={args.tuning} peft={args.peft} — loss (one batch)={loss}, eval_acc_sample={acc}")

if __name__ == "__main__":
    main()