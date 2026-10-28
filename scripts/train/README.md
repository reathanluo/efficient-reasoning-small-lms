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

Expected: 64 training examples, 16 validation examples, and three passing tests.
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
