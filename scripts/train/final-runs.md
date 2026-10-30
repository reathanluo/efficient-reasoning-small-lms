# Final training phase: sft-v1-20261031

The owned training configuration is frozen in `configs/sft-final.json`: both base
model sizes, FP32, rank 8 / alpha 16 / dropout 0.05, learning rate 0.0001, one
epoch, batch 1 / accumulation 8, maximum sequence length 4096, and seeds 0/1/2.
Every style uses the same 6547 ordered questions. Data, validation and prompt
source hashes are checked before execution. No test split is read or tuned on.
The earlier `sft-candidate.json` and provisional notes are historical records.

This common protocol was selected using human-data tuning and feasible full
teacher pilots; it is not proven optimal for each size or style. Do not continue
tuning each style to maximize its score. Any future protocol change needs a new
version and a review of comparability/reuse, rather than silently editing v1.

## Validation contract and speed

Final runs deliberately do NOT invoke the slow per-question HF validation loop.
They save training loss and the unmerged adapter first, exit that GPU process,
then run the existing vLLM generation/scoring modules on all 500 validation rows.
Until that succeeds, the training log says `trained_pending_validation`, with a
null validation accuracy rather than a fabricated number. On successful scoring,
the epoch record receives a vLLM accuracy/backend and status becomes `complete`.
Deferred validation is only implemented for the frozen one-epoch design.

Monitoring is uniformly greedy, seed 0, FP16 vLLM, 512 new tokens, max model
length 8192 for all styles and sizes. This cap is not a quality threshold: long
outputs often truncate. C's final accuracy-versus-budget/self-consistency study
remains separate and must include the agreed common budget grid. Do not compare
one condition's best-budget score against another's low-budget score.

Raw validation generations and scored/metadata/summary files are retained under
`runs/final_validation/<run-id>/<job-id>.*`. The corresponding per-epoch prediction
file is stored beside each adapter. The runner records final protocol/adapter
hashes, data hash, base model, actual adapter path and evaluation path in
`scripts/train/records/final_runs/<run-id>.json`. This does not automatically mark
weights backed up or handed off.

## First: verify the revised pipeline and register existing pilots

After pushing the Mac changes and pulling on TC1:

```bash
smoke_job=$(sbatch --parsable scripts/train/slurm_deferred_smoke.sh)
echo "Final pipeline check: $smoke_job"
sbatch --dependency=afterok:"$smoke_job" scripts/train/slurm_register_pilots.sh
```

These are two jobs, matching the documented submission limit. The small check
trains 64 human examples, saves with a pending-validation status, validates via
vLLM in a new process and tests that the score attaches correctly. It must print
`FINAL PIPELINE SMOKE PASSED` and exit 0. The GPU integration is not proven by
local tests alone. If it fails, investigate; do not bypass the dependency.

Registration verifies the full pilot logs, adapter configs, settings and data
hashes before reuse. It does NOT train the pilots again. The actual paths are:

- `long_15B_s0`: `ckpt/long_15B_s0_pilot_70806`
- `standard_05B_s0`: `ckpt/standard_05B_s0_pilot_70807`

It scores both with the common monitoring protocol, preserves old HF scores in
separate fields, and creates the two final-run records. Before these checks pass,
the pilots are reuse candidates, not two automatically completed final rows.
Do not rename/move their folders or point C to an invented canonical path.

## Then: train the remaining 22 entries

Submit one explicit matrix entry per job, at most one running and one queued:

```bash
sbatch scripts/train/slurm_final.sh concise 05B 0
sbatch scripts/train/slurm_final.sh answer_only 05B 0
```

Valid styles are long/standard/concise/answer_only, sizes 05B/15B and seeds 0/1/2.
Use `configs/final-run-checklist.csv` to avoid duplicates. No script automatically
submits all 24 jobs. To preview any entry without GPU/model work:

```bash
python -m reasoning.train.run_final concise 05B 0 --dry-run
```

Ordinary outputs use `ckpt/<style>_<size>_s<seed>`. Runs reserved for pilot reuse
will fail if the pilot isn't accessible; arrange an approved copy/shared-folder
location instead of accidentally duplicating seed-zero training.

For parallel team accounts, assign disjoint run IDs explicitly, use each person's
assigned QoS and confirm shared-folder access. The local manifest is a plan, not
proof of remote job status. Do not have two accounts train the same output path.
No messages or jobs on teammates' accounts have been sent/submitted by Codex.

## Failure and recovery

Every 100 training updates the runner saves optimizer/scheduler/RNG checkpoints;
the latest two are kept. Recover an interrupted training run with the same run ID:

```bash
sbatch scripts/train/slurm_final.sh long 05B 1 --resume-from-checkpoint ckpt/long_05B_s1/checkpoint-400
```

Use the actual latest COMPLETE checkpoint, not the example above blindly. Frozen
settings/data must match. Do not resume a numerical failure without investigation.
The earlier Trainer/PEFT save/resume integration check passed (70805); the new
deferred pipeline gets its own GPU check before final runs.

If training saved successfully but external evaluation failed/timed out, resubmit
the SAME style/size/seed without `--resume-from-checkpoint`. The runner verifies
the saved adapter and scores it without retraining. Existing incomplete/failed
training directories require explicit recovery; they are not overwritten.

## What the pilot results establish

- Full 1.5B long pilot: trained/saved, FP32, 5h15 including 2h39 HF validation,
  peak PyTorch allocation 13.499 GiB. Comparable vLLM scores at caps 512/2048/4096
  were 28.2%/64.6%/62.6%, with truncation 95.0%/30.8%/31.0% (70892).
- Full 0.5B standard pilot: trained/saved, FP32, about 92 minutes including 66
  minutes HF validation, peak 3.693 GiB. vLLM cap-512 score: 60.0% (70909).
- The different sizes/styles and diagnostic generation settings do not support
  a causal style comparison between these two pilots. Use the completed matrix.
- Training is unchanged by deferring validation, but frozen monitoring's uniform
  max model length can produce slightly different GPU answers from earlier runs.
  The consistent registration scores are recorded rather than copying old scores.

## Records and backups

Run `python -m reasoning.train.collect_records` on TC1 to archive small evidence.
Review and commit explicit record files. Copy weights separately to a Mac or an
approved shared folder and verify SHA-256 checksums. See
`scripts/train/progress-and-handoff.md` for commands. TC1 has no home backup.
Final Git records are evidence of training/evaluation, not proof of weight backup.
