# Provisional tuning plan (not frozen settings)

The completed development run used Qwen2.5-0.5B, human GSM8K solutions,
rank 8, learning rate 0.0002, one epoch, seed 0, effective batch size 8,
sequence limit 2048, and greedy validation capped at 512 new tokens.
It scored 253/500 (50.6%) on validation in about 60 minutes including training.
This is a development reference, not a teacher-data condition or final test result.

## Before tuning

Run the revised smoke job and verify its output, adapter and validation log.
Validation now enables KV caching, uses left-padded batches of eight, and records
progress. Compare batch-size 1 and 8 results on the same saved adapter before
relying on the new path. GPU numerical differences can change greedy outputs;
if differences are material, investigate them before comparing trial scores.
The unit tests use synthetic outputs; they do not establish real-model parity.
Use the same implementation, batch size and generation cap for tuning trials.

## Staged sweep

Use the full training split, all 500 validation questions, and seed 0 throughout.
Never read the test split for selection. Start with the 0.5B model.

1. Learning rate: 0.00005, 0.0001, 0.0002; rank 8 and one epoch.
2. Rank: 8, 16, 32 at the selected learning rate and one epoch.
3. Epoch budget: train the selected setting for three epochs and compare the
   per-epoch validation accuracies. The root adapter is from the last epoch;
   earlier adapters are in `checkpoint-*`. Check checkpoint availability before
   choosing an earlier epoch (only the latest two are retained).
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
support first; the current CLI does not resume a checkpoint.
