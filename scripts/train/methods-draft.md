# Person B Methods draft

Draft updated 1 November 2026 from the completed 24-run records and verified
Mac restore. The common protocol is `sft-v1-20261031` in `configs/sft-final.json`.
The deferred-validation integration check passed (70910), followed by pilot reuse
verification (70911). This text is prepared for the report but has not been
inserted into the shared document or independently reviewed by the team.

## Supervised fine-tuning

We investigate LoRA supervised fine-tuning of the Qwen2.5-0.5B and
Qwen2.5-1.5B base models on mathematical question-answer pairs. The teacher-data
handoff contains 6,547 examples in each of four conditions: long reasoning,
standard reasoning, concise reasoning, and answer-only. Conditions contain the
same ordered problem IDs, questions and gold answers. Teacher generation and
filtering are described in the Data subsection by Person A. Our training-side
checks verify file structure, training-split membership and matched examples;
they do not independently establish the correctness of every reasoning step.

Inputs use the shared tokenizer chat template and the fixed instruction from
`reasoning.prompts.chat_prompt`. Targets use `format_target`, which appends a
final line in the form `#### <number>` to the rationale. The answer-only target
contains this final-answer line without a rationale. We append an EOS token and
minimize autoregressive cross-entropy over completion tokens only. Prompt and
padding labels are masked with -100. Completion supervision includes reasoning,
the final answer and EOS; it is not limited to the numeric answer. Training uses
right padding and excludes no example silently based on its length.

## LoRA and training setup

The base model weights remain frozen. LoRA adapters are attached to the query,
key, value, output, gate, up and down projections. The frozen configuration
uses rank 8, alpha 16, dropout 0.05 and no trainable biases. We save unmerged PEFT
adapters so evaluation can load each adapter with its original base model.

The frozen protocol uses learning rate 1e-4, one Trainer-configured epoch, batch size one and
eight gradient-accumulation steps, giving an effective batch size of eight on
one GPU. The implementation uses Trainer's AdamW optimizer, linear learning-rate
decay, no warmup, zero weight decay and gradient checkpointing. Final logs record
818 optimizer updates per 6,547-row training file. Configurations and adapter
parameters were checked against the frozen file in the restored 24-run audit.
The learning-rate schedule spans each run's full epoch budget; the first epoch
of a three-epoch run is therefore not equivalent to an independent one-epoch run.

Jobs run through SLURM on an NVIDIA V100-PCIE with 32 GB GPU memory. The frozen
training precision is FP32 for both sizes. An FP16 1.5B smoke run produced NaN
gradient norms from its first logged step, and a three-epoch 0.5B FP16 run later
failed on a NaN gradient. Invalid runs are not treated as completed results.
FP32 smoke checks and all 24 final training runs completed successfully. This
does not establish that FP32 is optimal or identify the exact operation behind
the FP16 numerical failures. Non-finite loss and gradient checks remain enabled.

A training-side sequence audit measured the complete prompt, rationale, answer
and EOS. The longest sequence was 4,039 tokens on both student tokenizers;
708 long examples exceeded 2,048 tokens. We therefore use 4,096 as the
common training limit for every condition. An eight-example length-stratified
memory test including the longest example succeeded on both models, reaching
13.43 GiB (1.5B) and 9.151 GiB (0.5B) peak PyTorch allocated memory. These figures
are diagnostic PyTorch allocations, not complete device-memory usage or full-run
cost estimates. The full long/1.5B pilot took 5h15 including 2h39 sequential HF validation;
the standard/0.5B pilot took about 92 minutes including 66 minutes HF validation.
Final runs use separate vLLM validation, so these legacy totals must not be
reported as pure training cost. Recorded final-phase timings are below.

| Model | Style | Mean recorded training phase (min) | Mean vLLM validation (s) |
|---|---|---:|---:|
| 0.5B | Standard | 25.6 | 103.4 |
| 0.5B | Concise | 24.9 | 59.6 |
| 0.5B | Answer-only | 25.2 | 47.4 |
| 0.5B | Long | 68.4 | 138.9 |
| 1.5B | Standard | 42.7 | 123.2 |
| 1.5B | Concise | 29.7 | 67.0 |
| 1.5B | Answer-only | 30.1 | 50.4 |
| 1.5B | Long | 154.5 | 167.9 |

Means use three seeds. Training-phase time is the recorded training/save wall
time, with legacy sequential HF validation subtracted for the two reused pilots.
It excludes queue waits and does not measure all model loading/setup or GPU
billing time. vLLM time is recorded separately and includes its generation
initialization path. These are implementation-specific measurements, not
hardware-independent compute estimates.

## Hyperparameter selection and validation

Development used the 6,973 human-solution GSM8K training examples and the fixed
500-question validation split. We compared learning rates 5e-5, 1e-4 and 2e-4,
LoRA ranks 8, 16 and 32, and independent one- and two-epoch budgets. A three-epoch
attempt failed numerically and does not provide a completed three-epoch score.
Selection was staged rather than a complete Cartesian grid. All initial tuning
used seed zero on the 0.5B model in FP16; the later common FP32/teacher-data protocol was
confirmed feasible in two full teacher pilots. Tuning on human rationales may not find the best
settings for each teacher style. We report this limitation and use one
common frozen protocol for the final comparison rather than tune each style.

For comparable development scores we use the existing vLLM evaluator with the
same prompts, greedy decoding and a 512-new-token cap on all 500 questions.
Answers are extracted and scored by the shared numeric parser. One-epoch,
rank-8, LR-1e-4 training scored 51.4% (257/500), versus 43.0% (215/500) for the
matched untrained chat baseline. The baseline is the base model prompted in the
same chat format; it is not the project's separate eight-shot baseline.
These are validation observations, not final test results or significance claims.

A diagnostic using one saved adapter found 50.8% for single/no-cache HF
validation, 50.6% for single/cached HF, 46.8% for batch-eight/cached HF, and 50.8%
for vLLM. Single/no-cache and batch-eight differed in correctness on 24 questions.
The precise cause remains unresolved; equal aggregate accuracy does not prove
identical answers. Batch-eight HF scores were excluded from hyperparameter
ranking, and saved adapters were rescored using a consistent vLLM path. Historical development runs used single-question cached HF validation. Final
one-epoch runs save the adapter, release the training process, then use vLLM
validation; the score is attached to the epoch log with its backend identified.
A training save with outstanding validation is explicitly marked pending.
Final decoding caps and self-consistency settings belong to Person C's evaluation
protocol and must not be inferred from this development cap.

## Reproducibility, recovery and handoff

The environment snapshot records Python 3.11.18, PyTorch 2.5.1, Transformers
4.49.0, PEFT 0.14.0, Accelerate 1.4.0 and vLLM 0.7.3. Input SHA-256 hashes and
training arguments are recorded in new training logs. Long pilots checkpoint
adapter, optimizer, scheduler and RNG state every 100 updates and retain two
recent checkpoints. A TC1 integration check resumed a step-four checkpoint and
verified completed adapter weights against uninterrupted training within tolerance
(job 70805, completed successfully). An interrupted job can resume only with
matching recorded inputs and training settings.

We trained and validation-scored four styles x two sizes x three seeds = 24 adapters.
Seeds are fixed at 0, 1 and 2 for every style and size. Each handoff includes the unmerged
adapter path, base model, data hash, final config, train log and evaluation
artifacts. Keep the test split untouched during configuration selection.
The frozen configuration ID is `sft-v1-20261031`. Per-run IDs, actual adapter paths,
weight/data hashes and scores are in `scripts/train/records/final-handoff-manifest.csv`.
The two pilot-named seed-zero folders are preserved in that manifest rather than
renamed. All 24 restored weight hashes and all 12,000 validation correctness flags
were checked on 1 November. The weights were published as the GitHub Release
`person-b-adapters-20261101` on 1 November, and C evaluated all 24 adapters from
it on `gsm8k_val` and `gsm8k_test`.

## Frozen hyperparameter table

| Setting | Frozen value | Status |
|---|---|---|
| Students | Qwen2.5-0.5B / 1.5B base | Fixed research design |
| Training precision | FP32, both sizes | All 24 final runs completed |
| Training sequence limit | 4096 | Covers audited data; longest-example fit passed |
| Learning rate | 0.0001 | Frozen; selected using human-data validation |
| Epochs | 1 | Frozen |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 | Frozen |
| Target modules | q/k/v/o/gate/up/down projections | Implemented |
| Batch / accumulation | 1 / 8 | Frozen effective batch 8 |
| Final epoch validation | Separate vLLM process, greedy, 512-token cap | Common monitoring; no per-style cap selection |
| Comparable tuning scoring | vLLM FP16, greedy, 512-token cap | Implemented |
| Seeds | 0, 1, 2 | Fixed prospectively across all conditions |

## AI-use note for the team

Codex assisted with training code, tests, cluster-job scripts, debugging and this
draft. The author should review the implementation, validate the reported numbers
against archived artifacts, and incorporate the assistance into the team's AI Use
Statement. Do not claim that this draft or the final study has already been
independently reviewed.
