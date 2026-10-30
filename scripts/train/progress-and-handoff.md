# Person B progress and handoff

Snapshot: 30 October 2026 (historical; see the 31 October update below). This document records evidence reported from TC1;
Mac-local records alone do not prove remote job/filesystem state. This is not a
declaration that the final protocol or adapters are complete.

## Current update: 31 October

The authoritative protocol is frozen in `configs/sft-final.json` (seeds 0/1/2),
and current commands are in `scripts/train/final-runs.md`. Both full teacher
pilots completed and were evaluated. The 1.5B long model scored 28.2%, 64.6%
and 62.6% at 512/2048/4096 output caps; the 0.5B standard model scored 60.0%
at cap 512 through vLLM. These are different conditions and not a causal style
comparison. The earlier statuses below are historical.

Two pilots are reuse candidates pending programmatic config/hash checks and
consistent registration scoring. The new deferred-validation workflow must pass
a GPU smoke check before the remaining 22 training jobs. Code is prepared;
no new jobs, backups or team messages were submitted by Codex.

## Historical state before pilot completion

- Candidate: FP32, 4096-token limit, LR 0.0001, rank 8, alpha 16, one epoch,
  batch 1 / accumulation 8. Final seeds 0/1/2 are proposed, not confirmed.
- Teacher handoff: 6547 matched problems in all four styles. Full sequence max
  4039 for both sizes. Audit job 70801 completed.
- Small FP32 tests passed for 1.5B (70703) and 0.5B (70802). 1.5B vLLM adapter
  integration passed (70709), matching 13/16 answers in aggregate accuracy.
- Long-sequence FP32 fit passed for both sizes (70803, COMPLETED 0:0).
- Recovery integration passed (70805, COMPLETED 0:0, 1m18s).
- Full 1.5B/long pilot 70806 was last confirmed RUNNING. Its result is pending.
- A full 0.5B/standard pilot command was supplied, but submission/job ID has not
  been confirmed. Do not assume it is running or consume a queue slot blindly.
- No final 24-run matrix entries are marked complete. Pilots can be considered
  for reuse only after final settings, hashes, seeds and output requirements match.

## Files to keep current

- `configs/tuning-results.csv`: comparable validation observations and failures.
- `configs/final-run-checklist.csv`: proposed 24 entries, status and handoff paths.
- `configs/sft-candidate.json`: provisional settings and local data hashes; not a
  config currently consumed by the training CLI and not a frozen final config.
- `scripts/train/methods-draft.md`: report text to review and move to shared Doc.
- `scripts/train/records/`: small logs, eval summaries and audit snapshots collected
  from TC1; collect before committing so GitHub preserves the actual evidence.

## Finish the pilot stage

1. Read pilot 70806's `.out`, `.err`, `sacct` status, train log, peak memory and
   final saved adapter. Record full runtime and any retries.
2. Confirm the standard pilot's actual job ID/status, or submit it when a slot is
   free. Avoid duplicate full runs.
3. Score completed pilots through the same vLLM evaluation path. Check dataset
   hashes and all 500 validation questions; small smoke scores are not final scores.
4. Review full-run stability/runtime and freeze a documented common protocol.
   Do not reuse old FP16 human-data adapters as final FP32 teacher-data conditions.
5. Confirm final seeds and a shared folder that all three members are allowed to
   use. Assign disjoint run IDs before distributing jobs among team accounts.

## Final-run checklist workflow

Use the CSV rows as a tracker, not an automatic launcher. Suggested statuses:
`planned -> submitted -> running -> trained -> evaluated -> backed_up -> handed_off`.
Set `failed` for a failed attempt and retain its log; record a recovery attempt or
replacement job separately in notes. Include original and resume SLURM IDs.
Only update a status when the associated evidence exists.

For every completed adapter:

- Check `status: complete`, finite metrics, the intended base model/settings/data
  hashes, saved adapter weights/config, and per-epoch validation generations.
- Score through C's existing vLLM path; keep raw generations so scoring can be
  reproduced without another GPU run.
- Collect the small records and back up the adapter. TC1 explicitly provides no
  home-directory backup. Logging out preserves files but is not a backup.
- Hand C the shared-folder path, base model, frozen config ID, data SHA-256,
  seed, style, train/eval summary paths and any limitations.
- Do not put adapter weights, raw generations, optimizer states or model caches
  in Git. Final tables/figures belong to C; ask the owner before changing them.

## Record and back up small evidence

In the TC1 checkout after pulling the helper:

```bash
python -m reasoning.train.collect_records
git status
```

The helper copies finished/failed root training logs and small evaluation
summaries/metadata/audits into `scripts/train/records/`. It also creates an index
and file checksums. It does not copy large model weights or raw generations,
alter cluster jobs, publish messages, or update tracker statuses automatically.
Existing records are retained; changed source logs replace their record copies.
Files marked `running` are skipped.

Inspect the copied records, then commit explicit paths:

```bash
git add scripts/train/records configs/tuning-results.csv configs/final-run-checklist.csv
git commit -m "Archive Person B training and validation evidence"
git push origin b/partb
```

On the Mac, pull the branch to obtain these small records. Review the existing
`configs/tc1-smoke-packages.txt` snapshot as provenance; it includes a machine-local
`packaging @ file:///...` entry and is not a portable one-command requirements file.

## Back up adapter weights to the Mac

Run these on the Mac's LOCAL terminal while connected to NTU network/VPN, not
inside SSH. Example for the long-fit diagnostics (completed):

```bash
mkdir -p ~/Downloads/sc4001-backups
scp -r jchua134@10.96.189.11:~/sc4001-group-project/ckpt/long_fit_1.5B_70803 ~/Downloads/sc4001-backups/
scp -r jchua134@10.96.189.11:~/sc4001-group-project/ckpt/long_fit_0.5B_70803 ~/Downloads/sc4001-backups/
```

For the full pilot, wait for completion and use its actual adapter directory.
`scp -r` preserves the cluster copy and includes any nested checkpoints, which
may make the transfer larger. Keep complete Trainer checkpoints if recovery from
another computer/storage location is required; adapter weights alone cannot
restore optimizer/scheduler/RNG state.

After downloading an adapter, compare the SHA-256 of its weights. On TC1:

```bash
sha256sum ckpt/long_fit_1.5B_70803/adapter_model.safetensors
```

On the Mac:

```bash
shasum -a 256 ~/Downloads/sc4001-backups/long_fit_1.5B_70803/adapter_model.safetensors
```

The hashes must match. Record verified backup location/date in the checklist.
Backups and pushes described here have NOT been performed by Codex.

## Draft handoff message (not sent)

Person B update: the training pipeline, dataset audit, longest-example FP32 fit
for both student sizes, and checkpoint-recovery test have passed. The candidate
is LR 1e-4, rank 8, one epoch, FP32 and a 4096-token limit. The full 1.5B long
teacher-data pilot is running (70806); final settings are not frozen. Comparable
human-data validation was 51.4% for the leading 0.5B adapter versus 43.0% for a
matched untrained chat baseline. These are validation results, not final tests.
Next: confirm full teacher-run stability, freeze the protocol, then distribute
the 24-run matrix. Proposed seeds are 0, 1 and 2; shared-folder handoff paths
still need confirming.
