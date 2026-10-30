# Training job scripts (owner: Person B)

**30 October 1.5B diagnostic:** job 70517 logged NaN gradient norms at all
eight steps, zero logged loss, no learning-rate decay, and 0/16 validation.
Do not use its adapter as a trained result. Trainer's default non-finite loss
logging filter can hide invalid losses; the script now disables that filter,
rejects non-finite loss/gradient metrics and records `status: failed` on training
exceptions. FP16 instability is a hypothesis, not an established root cause.
`sbatch scripts/train/slurm_smoke15_fp32.sh` repeats the same tiny setup in FP32,
including FP32 base weights and disabled FP16 mixed precision. This requires more
memory and may run slower. Do not freeze precision choices until the diagnostic
and a comparable 0.5B check pass; report any precision difference across sizes.

**29 October validation audit:** do not use the batch-eight training-validation
scores for hyperparameter selection. Job 70493 scored the same adapter at 50.8%
with single/no-cache HF, 50.6% with single/cached HF, 46.8% with batch-eight HF,
and 50.8% with vLLM. The root cause remains unverified. Defaults now use batch
size one. The batching description later in this document records the earlier
implementation, not the current default. Run `sbatch scripts/train/slurm_rescore_trials.sh`
to score the four saved trial adapters and a matched untrained baseline using
the same vLLM protocol. This job does no training. See `configs/tuning-plan.md`.

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
masking tests, one dependency-free comparison test, and three validation tests
that require torch (seven tests in total).
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

## Investigate validation differences before selecting a winner

The completed tuning trials scored 47.0% (LR 0.00005, rank 8), 47.2%
(LR 0.0001, rank 8), 46.8% (LR 0.0002, rank 8), and 47.4%
(LR 0.0002, rank 16). Differences this small do not establish a robust winner.
The old validation path scored 50.6% for LR 0.0002/rank 8, so first isolate the
validation changes on a single saved adapter rather than retraining.

```bash
sbatch scripts/train/slurm_check_validation.sh
```

This defaults to the original `ckpt/human_05B_s0_full_e1` adapter and all 500
validation questions. It runs single-question/no-cache, single-question/cached,
batch-eight/cached HF generation, then the existing vLLM evaluator. All modes
use the same adapter, prompts, question order, greedy decoding and 512-token cap.
It saves generations, disagreement counts/IDs, accuracies and timing under
`runs/validation_check_<jobid>/`; see `summary.json` and the SLURM output for
the vLLM score. The first HF mode is compared with the adapter's saved epoch-1
predictions when present. No adapter is changed and no training is performed.
Expect a longer diagnostic job because it deliberately repeats slow validation.
The code passes local unit tests but this GPU comparison still needs a TC1 run.

For a shorter diagnostic first, pass an adapter and sample count explicitly:

```bash
sbatch scripts/train/slurm_check_validation.sh ckpt/human_05B_s0_full_e1 32
```

A 32-question agreement check cannot rule out differences elsewhere in the full
validation set. Do not freeze hyperparameters until the discrepancy is understood.
