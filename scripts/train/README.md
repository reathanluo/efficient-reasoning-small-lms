# Training job scripts (owner: Person B)

SLURM scripts that train the LoRA adapters into `ckpt/`. See `reasoning/train/__init__.py` for the output format.

## Start here: one smoke run

The first goal is to train on 64 human solutions, save an adapter, and load it
through the existing evaluation code. This checks the pipeline; its accuracy is
not a research result. The defaults are provisional, not frozen hyperparameters.

### 1. Check the data and masking locally

From the repository root (no GPU, downloads or extra packages required):

```bash
python3 -m reasoning.train.sft --check-data --limit 64 --val-limit 16
python3 -m unittest reasoning.train.test_sft -v
```

Expected: 64 training examples and 16 validation examples. There are three
dependency-free masking tests and three validation tests that require torch.
The tests cover prompt masking, answer-only targets, preserving EOS when padding
uses the same token, and refusing to truncate the answer.

### 2. Set up TC1

Follow the root README's cluster setup: NTU network/VPN, SSH, the `sc4001-g`
conda environment, `pip install -r requirements.txt`, and `MyTCinfo` to confirm
your assigned QoS. Use the pinned cluster versions; the laptop environment may
have incompatible versions. Cache the student model on the head node:

```bash
export HF_HOME="$HOME/hf_cache"
huggingface-cli download Qwen/Qwen2.5-0.5B
```

Transfer your Person B changes to the cluster checkout before submitting. Training
runs on a GPU through `sbatch`; do not train on the head node.

### 3. Submit the smoke job on TC1

```bash
sbatch scripts/train/slurm_smoke.sh
squeue -u <ntu-id>
tail -f slurm-b-sft-smoke-<jobid>.out
```

The script trains Qwen2.5-0.5B with rank 8, FP16, batch size 1, gradient
accumulation 8, and gradient checkpointing. It evaluates 16 validation questions
greedily after the epoch and saves:

```text
ckpt/human_05B_s0_smoke_<jobid>/
  adapter_config.json
  adapter_model.safetensors
  train_log.json
  validation_epoch_1.jsonl
```

There will also be tokenizer files and a resumable Trainer checkpoint. The final
adapter is unmerged. Check `status: complete`, finite training loss, and the
validation predictions in the log files. A small smoke run can have low accuracy.
Establishing decreasing loss requires looking at the step losses in `history`;
it is not guaranteed after this short run.

### 4. Confirm the adapter loads in evaluation

In a GPU SLURM job using the same environment, run these commands with the actual
adapter directory (the existing baseline SLURM script is an environment example):

```bash
python -m reasoning.eval.generate --model Qwen/Qwen2.5-0.5B \
    --lora ckpt/human_05B_s0_smoke_<jobid> --prompt chat \
    --split gsm8k_val --limit 16 --dtype float16 \
    --out runs/b_smoke_val.jsonl
python -m reasoning.eval.evaluate runs/b_smoke_val.jsonl
```

This is the remaining integration check after training. Do not treat successful
local tests as confirmation that the V100 training or vLLM adapter loading works.

## After the smoke run

1. Expand human-solution training and tune learning rate, epochs and rank using
   the full 500-question validation set. Remove `--limit` and `--val-limit` for
   those runs. Each run needs a fresh output directory.
2. Save the chosen settings in `configs/` and freeze them across styles, sizes
   and seeds. Log the sweep results and selection rule for the report.
3. When A's data arrives, change `--train-file` to `data/sft/<style>.jsonl` and
   `--out` to `ckpt/<style>_<size>_s<seed>`. The script automatically reads
   `rationale` instead of `solution`. Train the 24 combinations.

All training targets use the shared prompt/target helpers; only completion
tokens (rationale, final answer, EOS) contribute to loss. Input IDs are checked
against the committed training split. Overlength examples cause an explicit
error, rather than silently removing the answer or changing the problem set.
Before final runs, check all four styles against one common sequence limit that
fits the GPU. The initial 2048-token limit is a smoke-run starting point.

`train_log.json` records the CLI config, sample counts, epoch mean step losses,
validation accuracy, training history and elapsed time. Validation generations
are retained separately. The saved root adapter is from the final epoch;
choose the epoch budget on validation during tuning.

## Faster validation and first tuning trials

Validation uses `--val-batch-size 8` by default, left padding and generation KV
caching. It prints progress after each batch and updates `train_log.json` with
`stage` and `validation_progress`. The training collator still right-pads and
masks prompt/padding labels. Validation restores model mode and tokenizer padding
after generation, including on failure. The generation cap remains 512 tokens.
The log records validation duration per epoch. GPU speed and real-model numerical
parity still need verification on TC1; use `--val-batch-size 1` as a comparison.
If GPU memory runs out, reduce this batch size consistently across tuning runs.

First rerun the smoke job on the revised code (it uses a new job-specific output):

```bash
sbatch scripts/train/slurm_smoke.sh
```

Then start the staged sweep described in `configs/tuning-plan.md`. For example,
these are two independent learning-rate trials, submitted separately:

```bash
sbatch scripts/train/slurm_trial.sh 0.00005 8 1
sbatch scripts/train/slurm_trial.sh 0.0001 8 1
```

Arguments are learning rate, rank and epochs. Each trial uses full human-solution
training and full validation, and saves to `ckpt/human_05B_s0_trial_<jobid>/`.
Submit at most two jobs at a time under the documented QoS. After one finishes,
the third learning-rate trial is `sbatch scripts/train/slurm_trial.sh 0.0002 8 1`.
This repeats the reference setting under the revised validation implementation.
Only proceed to rank/epoch selection after inspecting the learning-rate results.

Do not reinstall the unbounded requirements over a working environment. The
recorded TC1 environment is `configs/tc1-smoke-packages.txt`; the working data
stack was NumPy 1.26.4, PyArrow 19.0.1 and datasets 3.3.2. These are setup notes,
not a change to the shared requirements file.
