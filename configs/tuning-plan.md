# Provisional tuning plan (not frozen settings)

**Superseded on 31 October:** this file retains the development history.
The common protocol is now frozen in `sft-final.json` using seeds 0/1/2.
See `scripts/train/final-runs.md` for the final-phase commands and required GPU
pipeline check. The two full pilots are eligible for reuse only after their
saved config/data/adapter checks pass on TC1. Do not treat historical pending
statuses below as live job state.

## Current status, 30 October

See `tuning-results.csv` for comparable scores and failures, `sft-candidate.json`
for the provisional FP32/4096 settings and data hashes, and
`final-run-checklist.csv` for the 24 planned runs. None of these declares the
configuration frozen. Proposed seeds 0/1/2 still need confirmation.

The leading human-data setting is LR 0.0001, rank 8, one epoch (51.4% on all 500
validation questions in vLLM). Independent two epochs scored 49.8%; the three-epoch
FP16 attempt failed numerically and has no final score. FP32 smoke checks and
longest-example fits passed for both sizes. Full 1.5B/long pilot 70806 was last
confirmed running; the standard 0.5B pilot submission is not confirmed.
No further exhaustive human-solution sweep is currently planned.

## Validation audit, 29 October (job 70493)

On the same original adapter and all 500 questions: single/no-cache HF scored
50.8%, single/cached HF 50.6%, batch-eight/cached HF 46.8%, and vLLM 50.8%.
Batch-eight changed correctness on 24 questions relative to single/no-cache;
single/cached changed correctness on five. Equal aggregate accuracies do not
establish per-question agreement. The precise source of the numerical/output
differences has not been established. Do not use batch-eight scores to select
hyperparameters or assume the existing trial ranking survives rescoring.

Training validation now defaults to batch size one. Use the existing vLLM
evaluation path with a fixed 512-token cap to rescore ALL four saved trials and
a matched untrained chat baseline, without retraining:

```bash
sbatch scripts/train/slurm_rescore_trials.sh
```

The earlier batch-eight plan below is historical; this audit supersedes its
batch-size guidance. Single-question HF validation is slow; budget accordingly.

The completed development run used Qwen2.5-0.5B, human GSM8K solutions,
rank 8, learning rate 0.0002, one epoch, seed 0, effective batch size 8,
sequence limit 2048, and greedy validation capped at 512 new tokens.
It scored 253/500 (50.6%) on validation in about 60 minutes including training.
This is a development reference, not a teacher-data condition or final test result.

## Before tuning

Training-time validation uses single-question cached HF generation. Use the same
vLLM implementation, prompt and generation cap to compare saved trial adapters;
do not mix HF and vLLM scores in the selection table. The batch-eight diagnostic
found material differences; its scores are not used for selection. The exact
cause remains unresolved. Local tests do not establish GPU numerical parity.

## Staged sweep

Use the full training split, all 500 validation questions, and seed 0 throughout.
Never read the test split for selection. Start with the 0.5B model.

1. Learning rate: 0.00005, 0.0001, 0.0002; rank 8 and one epoch.
2. Rank: 8, 16, 32 at the selected learning rate and one epoch.
3. Epoch budget: compare independent one- and two-epoch runs through vLLM.
   Record the three-epoch numerical failure separately; it is not a valid final
   three-epoch result. Epoch budgets have different learning-rate schedules.
4. Confirm the candidate fits and behaves sensibly on 1.5B and all four teacher
   styles. Use one common sequence limit and identical problem IDs across styles.
   Human-solution tuning need not be optimal for teacher traces; document that
   limitation and agree the final protocol with the team before freezing.

Select by validation accuracy. For exact ties, prefer lower rank and fewer epochs;
use the lower learning rate as the last tie breaker. Save a comparison table with
job IDs, output paths, settings, validation accuracy and timing. Freeze the chosen
settings in a separate config only after completing the selection procedure.

Submit one trial per job with `scripts/train/slurm_trial.sh`. The GPU allocation
allows one GPU and at most two jobs at once, so submit at most one running and one
pending trial. These jobs are independent; do not submit a dependent trial before
choosing its settings. Each job requests six hours. Longer runs need tested resume
support first. New hashed runs support guarded recovery; the TC1 save/resume
integration check passed in job 70805. See the training README for recovery.
