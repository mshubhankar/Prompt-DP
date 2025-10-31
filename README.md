# PROMPT-DP

## Overview
This repository uses a Conda environment specified in `environment.yml` and a wrapper script `script.sh` to run the code.

## Requirements
- Conda (or Mamba)
- `environment.yml` present in the repository root
- `script.sh` present and executable (or run with `bash`)
- uses private_transformers library from https://github.com/lxuechen/private-transformers

## Create the Conda environment (Python 3.9)
1. Ensure `environment.yml` requests Python 3.9. In `environment.yml` include:
    ```yaml
    name: myenv
    dependencies:
      - python=3.9
      # ... other dependencies ...
    ```
2. Create the environment:
    ```bash
    conda env create -f environment.yml
    ```
    Or with mamba:
    ```bash
    mamba env create -f environment.yml
    ```

If `environment.yml` does not specify a name, specify one when creating:
```bash
conda env create -n myenv -f environment.yml
```


## Activate the environment
```bash
conda activate myenv
```
(Replace `myenv` with the name in `environment.yml`.)

## Run the code
To run a single experiment, you can directly call the Python script with desired arguments. For example:
```bash
python private_batch.py --dataset sst2 --tuning full --peft lora --epochs 30 --use_dp
```

The current options for `--dataset` are `sst2` and `qnli`.

Or you may use the batch script that runs different types of experiments.
Make the script executable and run it, or run it with bash:
```bash
chmod +x script.sh
./script.sh
# or
bash script.sh
```
Always activate the Conda environment before running the script unless the script handles activation.
