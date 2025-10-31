import argparse
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import evaluate
from peft import get_peft_model, LoraConfig, TaskType, IA3Config
from private_transformers import PrivacyEngine
import torch.nn as nn

# CLI args
parser = argparse.ArgumentParser(description="DP prompt/PEFT training for tinyBERT")
parser.add_argument("--tuning", type=str, default="full", choices=["soft", "full", "prefix", "last"])
parser.add_argument("--peft", type=str, default=None, choices=[None, "lora", "ia3"])
parser.add_argument("--use_dp", action="store_true", help="enable differential privacy (private_transformers)")
parser.add_argument("--target_epsilon", type=float, default=8.0)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--promptlength", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-3)
args = parser.parse_args()

tuning = args.tuning  # options: "soft" or "full" or "prefix" or "last"
peft = args.peft  # options: "lora" or "ia3"

# Differential privacy config
use_dp = args.use_dp
target_epsilon = args.target_epsilon

# Load the SST-2 dataset
dataset = load_dataset("glue", "qnli")
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare data for PyTorch
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Load the model
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

# If using LoRA, wrap the model now (before any forward monkeypatching)
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

if peft == "ia3":

    ia3_config = IA3Config(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=["query", "value"]
    )

    model = get_peft_model(model, ia3_config)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
# criterion expects per-example loss for private engines in some flows
criterion = torch.nn.CrossEntropyLoss(reduction="none")

# --- moved device creation up so we can create the soft prompt on the same device ---
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model.to(device)


num_epochs = args.epochs
batch_size = args.batch_size
prompt_length = args.promptlength
lr = args.lr

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

metric = evaluate.load("accuracy")

if tuning == "last":
    # Freeze all layers except the last classifier layer
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Build optimizer from trainable params (only the classifier layer)
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

if tuning == "soft":
    # Replace bare Parameter with a small nn.Embedding-based module so private_transformers/opacus
    # can compute per-sample gradients for the prompt.
    class SoftPrompt(nn.Module):
        def __init__(self, prompt_length: int, hidden_size: int, device=None):
            super().__init__()
            self.prompt_length = prompt_length
            self.embedding = nn.Embedding(prompt_length, hidden_size)
            if device is not None:
                self.to(device)

        def forward(self, batch_size: int, device=None):
            dev = device if device is not None else self.embedding.weight.device
            # indices shape [batch, prompt_length] -> embedding lookup yields [batch, prompt_length, hidden]
            idx = torch.arange(self.prompt_length, device=dev).unsqueeze(0).expand(batch_size, -1)
            return self.embedding(idx)

    hidden_size = model.config.hidden_size

    # create and register the module on the embeddings submodule (so its params are under model.named_parameters())
    soft_prompt_module = SoftPrompt(prompt_length, hidden_size, device=device)
    model.bert.embeddings.add_module("soft_prompt", soft_prompt_module)

    # Freeze everything except LoRA params and the soft prompt params
    for name, param in model.named_parameters():
        if ("lora" in name.lower()) or ("soft_prompt" in name) or ("ia3" in name.lower()):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Rebuild optimizer from trainable params (do this AFTER registering the module and toggling requires_grad)
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found for 'soft' tuning — expected soft_prompt or LoRA params.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # Patch forward to use the module (per-sample lookup)
    original_forward = model.forward
    def new_forward(*args, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Accept input_ids or inputs_embeds (some privacy wrappers pass inputs_embeds)
        device_local = None
        if len(args) > 0 and input_ids is None and inputs_embeds is None:
            possible = args[0]
            if isinstance(possible, torch.Tensor):
                input_ids = possible

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("new_forward: neither input_ids nor inputs_embeds provided")
            inputs_embeds = model.bert.embeddings(input_ids=input_ids)
            device_local = input_ids.device
        else:
            device_local = inputs_embeds.device

        batch_size = inputs_embeds.size(0)
        # use the registered module so Opacus/private_transformers can hook it
        soft_prompt_expanded = model.bert.embeddings.soft_prompt(batch_size, device=device_local)
        inputs_embeds = torch.cat([soft_prompt_expanded, inputs_embeds], dim=1)

        # attention/token_type handling
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=device_local)
        else:
            prefix_mask = torch.ones((batch_size, prompt_length), dtype=attention_mask.dtype, device=device_local)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=device_local)
        else:
            prefix_token_type = torch.zeros((batch_size, prompt_length), dtype=token_type_ids.dtype, device=device_local)
            token_type_ids = torch.cat([prefix_token_type, token_type_ids], dim=1)

        dummy_input_ids = None
        if input_ids is None:
            dummy_input_ids = torch.zeros((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=device_local)

        return original_forward(input_ids=dummy_input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, **kwargs)
    model.forward = new_forward

if tuning == "prefix":
    hidden_size = model.bert.embeddings.word_embeddings.embedding_dim

    # Create an nn.Embedding per layer (not a bare Parameter) so Opacus can record grad_sample
    for idx, layer in enumerate(model.bert.encoder.layer):
        emb = nn.Embedding(prompt_length, hidden_size)
        # init similarly to your previous random init
        nn.init.normal_(emb.weight, mean=0.0, std=hidden_size ** -0.5)
        # move the embedding to the training device so lookups run on the same device as inputs
        emb = emb.to(device)
        # register the embedding module on the layer so it's visible as a submodule/parameter
        # use add_module to ensure proper registration under NamedModules/Parameters
        layer.add_module("prefix_emb", emb)

    # Freeze all params except LoRA params and the prefix embedding params
    for name, param in model.named_parameters():
        if ("lora" in name.lower()) or ("prefix_emb" in name) or ("ia3" in name.lower()):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Rebuild optimizer from trainable params (LoRA + prefix embeddings)
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found for 'prefix' tuning — expected prefix_emb or LoRA params.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # Modify forward to call each layer's embedding lookup (per-sample) instead of expanding a bare Parameter
    original_forward = model.forward
    def new_forward(input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        unused_kwargs = kwargs
        batch_size = input_ids.size(0)
        device_local = input_ids.device

        inputs_embeds = model.bert.embeddings(input_ids=input_ids)

        # ensure 2D attention_mask is long and on the right device
        if attention_mask is None:
            base_attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=device_local)
        else:
            base_attention_mask = attention_mask.to(dtype=torch.long, device=device_local)

        if token_type_ids is None:
            token_type_ids = torch.zeros((batch_size, inputs_embeds.size(1)), dtype=torch.long, device=device_local)
        else:
            token_type_ids = token_type_ids.to(dtype=torch.long, device=device_local)

        hidden_states = inputs_embeds
        for i, layer in enumerate(model.bert.encoder.layer):
            # per-batch lookup from the registered nn.Embedding module
            idx = torch.arange(prompt_length, device=device_local).unsqueeze(0).expand(batch_size, -1)
            prefix_emb = layer.prefix_emb(idx)  # shape [batch, prompt_length, hidden_size]
            hidden_states = torch.cat([prefix_emb, hidden_states], dim=1)

            prefix_mask = torch.ones((batch_size, prompt_length), dtype=base_attention_mask.dtype, device=device_local)
            base_attention_mask = torch.cat([prefix_mask, base_attention_mask], dim=1)

            extended_attention_mask = model.bert.get_extended_attention_mask(base_attention_mask, (batch_size, hidden_states.size(1)), device_local)

            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]

        pooled_output = model.bert.pooler(hidden_states)
        logits = model.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = criterion(logits, labels)

        from types import SimpleNamespace
        if loss is None:
            return SimpleNamespace(logits=logits)
        else:
            return SimpleNamespace(loss=loss, logits=logits)
    model.forward = new_forward

# After applying PEFT (get_peft_model) and after model.to(device),
# ensure any wrapper.original_module modules are registered as real submodules
# so Opacus/private_transformers can install hooks on them.

# build a lookup for modules by name
_name2module = dict(model.named_modules())

for mod_name, mod in list(_name2module.items()):
    orig = getattr(mod, "original_module", None)
    if orig is None:
        continue
    # determine parent module to attach the original under
    if "." in mod_name:
        parent_name, child_name = mod_name.rsplit(".", 1)
        parent = _name2module.get(parent_name, model)
    else:
        parent = model
        child_name = mod_name

    # create a safe unique name
    alias_name = f"{child_name}_original"
    # avoid clobbering existing attributes/modules
    if not hasattr(parent, alias_name):
        # register the original module as a submodule of the parent
        parent.add_module(alias_name, orig)
        # also update lookup so further iterations can find it
        _name2module[f"{parent_name}.{alias_name}" if "." in mod_name else alias_name] = orig

# Ensure backbone params are frozen by default; enable only adapter/prompt params
if tuning != "full":
    for n, p in model.named_parameters():
        # keep LoRA/IA3 adapter params and any prompt/prefix params trainable, freeze the rest
        lower = n.lower()
        if ("lora" in lower) or ("ia3" in lower) or ("soft_prompt" in n) or ("prefix_emb" in n) or (tuning == "last" and "classifier" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

 # print number of trainable params
total_params = sum(p.numel() for p in model.parameters())
if tuning == "full":
    trainable_params = list(model.parameters())
trainable_count = sum(p.numel() for p in trainable_params)

print(f"Total params: {total_params}, Trainable params: {trainable_count} ({100.0 * trainable_count / total_params:.2f}%)")

# Rebuild optimizer after registering these aliases and after any requires_grad toggles
trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise ValueError("No trainable parameters found after registering original modules")
optimizer = torch.optim.AdamW(trainable_params, lr=5e-3)


# Differential Privacy: attach opacus PrivacyEngine after final optimizer selection and before training loop
if use_dp:
    
    # make_private returns (model, optimizer, data_loader)
    privacy_engine = PrivacyEngine(
    model,
    batch_size=batch_size,
    sample_size=len(train_dataset),
    epochs=num_epochs,
    max_grad_norm=0.1,
    target_epsilon=target_epsilon,
    )
    privacy_engine.attach(optimizer)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # pop/move labels first (so Opacus sees consistent batch shape)
        labels = batch.pop("label")
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device)

        # move remaining tensors to device (keep any Opacus-added fields)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(outputs.logits, labels)
        # loss.backward()
        optimizer.step(loss=loss)
        
    model.eval()
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label") # Remove 'label' from the batch
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels) # Use the separated labels for metric calculation
    print(f"Epoch {epoch+1} Loss: {loss.mean().item()}")
    print(f"Epoch {epoch+1} Accuracy: {metric.compute()}")
print(f'Tuning method: {tuning}, PEFT method: {peft}, Target Epsilon: {target_epsilon}')
print("Training finished!")