#!/bin/sh
python nonprivate_batch.py --tuning soft  --epochs 30 --promptlength 100
python nonprivate_batch.py --tuning prefix  --epochs 30 --promptlength 100
python nonprivate_batch.py --tuning full --peft lora --epochs 30
python nonprivate_batch.py --tuning soft --peft lora --epochs 30 --promptlength 100
python nonprivate_batch.py --tuning prefix --peft lora --epochs 30 --promptlength 100
python nonprivate_batch.py --tuning last  --epochs 30
python nonprivate_batch.py --tuning full  --epochs 30
python nonprivate_batch.py --tuning full --peft ia3 --epochs 30


# python private_batch.py --tuning last  --epochs 30 --use_dp
# python private_batch.py --tuning full  --epochs 30 --use_dp
# python private_batch.py --tuning soft  --epochs 30 --promptlength 100 --use_dp --lr 5e-4
# python private_batch.py --tuning prefix  --epochs 30 --promptlength 100 --use_dp --lr 5e-4
# python private_batch.py --tuning full --peft lora --epochs 30 --use_dp
# python private_batch.py --tuning soft --peft lora --epochs 30 --promptlength 100 --use_dp
# python private_batch.py --tuning prefix --peft lora --epochs 30 --promptlength 100 --use_dp
# python private_batch.py --tuning full --peft ia3 --epochs 30 --use_dp