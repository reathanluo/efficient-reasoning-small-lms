# TC1 offline runbook: remaining Person B jobs

Saved on the Mac so you can read/copy commands while Codex is unavailable over VPN.
Open this file in a local text editor before connecting. No new training code is
required for the job pairs below. All commands in sections 1-7 run INSIDE TC1 SSH;
backup section 8 explicitly separates cluster commands from Mac-local commands.

## 1. Connect from Mac Terminal

Connect to NTU network/VPN first. Run on your Mac:

```bash
ssh jchua134@10.96.189.11
```

Enter the NTU password (typing is invisible), then run separately on TC1:

```bash
cd ~/sc4001-group-project
module load anaconda
source activate sc4001-g
python --version
```

Expected Python: 3.11.18. Do not reinstall packages, change the frozen settings,
or run model training directly on the head node. Jobs use `sbatch` below.

## 2. What is already done / underway

Confirmed trained/evaluated before the current long-0.5B pair:

- `standard_05B_s0`, `standard_05B_s1`, `standard_05B_s2`.
- `concise_05B_s0`, `concise_05B_s1`, `concise_05B_s2`.
- `answer_only_05B_s0`, `answer_only_05B_s1`, `answer_only_05B_s2`.
- `long_15B_s0` (registered pilot; do NOT train this again).

That is 10/24 evaluated adapters. The current pair is `long_05B_s0` and
`long_05B_s1`; submission IDs were not reported in chat. After BOTH complete,
the count should be 12/24. Do not resubmit either without checking records/logs.

The full matrix has four styles, two sizes (05B/15B), seeds 0/1/2. Frozen settings
are read by the launcher from `configs/sft-final.json`. Keep the same settings for
all conditions, including low-scoring answer-only runs.

## 3. Check the current jobs and actual completion

```bash
squeue -u "$USER"
```

One RUNNING plus one PENDING is normal. `QOSMaxGRESPerUser` means your one GPU
is occupied. `QOSMaxSubmitJobPerUserLimit` means a new submission was REJECTED;
it did not create a job. Never repeat a successful submission just because it
is pending. Note the ID from every `Submitted batch job ...` message.

All completed score lines (includes the two registered pilots):

```bash
grep 'FINAL RUN COMPLETE' slurm-b-final-*.out slurm-b-register-*.out
```

For the pair currently underway, look specifically for `long_05B_s0` and
`long_05B_s1`. A final record is stronger evidence than merely leaving the queue:

```bash
python -c 'import glob,json; r=[json.load(open(p)) for p in glob.glob("scripts/train/records/final_runs/*.json")]; d={x["run_id"]:x for x in r if x.get("status")=="evaluated" and x.get("protocol_id")=="sft-v1-20261031"}; print("Evaluated:",len(d),"/24"); [print(k,"accuracy:",d[k]["accuracy"],"adapter:",d[k]["adapter"]) for k in sorted(d)]'
```

For one job, replace JOBID with its actual number:

```bash
tail -n 15 slurm-b-final-JOBID.out
tail -n 25 slurm-b-final-JOBID.err
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

For two actual job numbers, use `sacct -j NUMBER1,NUMBER2 ...`. A successful job
should show COMPLETED, ExitCode 0:0, `FINAL RUN COMPLETE` and an evaluated record.
If `squeue` says invalid job ID or no row, use `sacct`; do not assume success.

## 4. The next two commands AFTER the current pair completes

Check that your queue is empty and the current long seed-0/seed-1 records exist.
If the following entries have not already been submitted/completed, run separately:

```bash
sbatch scripts/train/slurm_final.sh long 05B 2
sbatch scripts/train/slurm_final.sh standard 15B 0
```

After these finish successfully: 14/24 evaluated. Record their job IDs.

## 5. The rest of the pairs, in order

Run only ONE pair at a time, after the preceding pair finishes. If you choose
to refill a single vacant slot earlier, keep at most TWO submitted jobs total
(running + queued), and cross off what you submit so you do not duplicate it.

### Pair B: standard 1.5B seeds 1 and 2

```bash
sbatch scripts/train/slurm_final.sh standard 15B 1
sbatch scripts/train/slurm_final.sh standard 15B 2
```

After completion: 16/24 evaluated.

### Pair C: concise 1.5B seeds 0 and 1

```bash
sbatch scripts/train/slurm_final.sh concise 15B 0
sbatch scripts/train/slurm_final.sh concise 15B 1
```

After completion: 18/24 evaluated.

### Pair D: concise 1.5B seed 2 and answer-only 1.5B seed 0

```bash
sbatch scripts/train/slurm_final.sh concise 15B 2
sbatch scripts/train/slurm_final.sh answer_only 15B 0
```

After completion: 20/24 evaluated.

### Pair E: answer-only 1.5B seeds 1 and 2

```bash
sbatch scripts/train/slurm_final.sh answer_only 15B 1
sbatch scripts/train/slurm_final.sh answer_only 15B 2
```

After completion: 22/24 evaluated.

### Pair F: long 1.5B seeds 1 and 2

Seed 0 is ALREADY DONE; do not repeat it.

```bash
sbatch scripts/train/slurm_final.sh long 15B 1
sbatch scripts/train/slurm_final.sh long 15B 2
```

After completion: 24/24 trained/evaluated, assuming all earlier entries succeeded.
This does not yet mean all weights have been backed up/handed to C or the report
is finished. Long targets take longer. Training/validation now run in separate
processes, so the old hours-long HF validation loop is not repeated.

## 6. Stop watching / disconnect / stop a job

`tail -f FILE` watches for new log lines; it can sit silently without a prompt.
Ctrl+C stops watching; it does NOT cancel the submitted job.

To disconnect from TC1 safely:

```bash
exit
```

SLURM continues submitted jobs after disconnecting. You can close the laptop.
You do NOT need to keep Terminal or Codex open. Use `scancel JOBID` only when
you intentionally want to cancel that specific job, not merely disconnect.

## 7. If a run fails: do NOT blindly resubmit or change precision/settings

Inspect its `.err`, `.out`, `sacct` status and `ckpt/<run-id>/train_log.json`.
The two pilot folders have special names recorded in the final-run records.

- `status: trained_pending_validation`: training saved, but external vLLM scoring
  has not completed. Resubmit the SAME style/size/seed command. The launcher
  verifies and reuses the adapter, then scores it without retraining.
- `status: complete` but no final record: inspect the error and whether generation
  or final registration failed. The same run command can reuse a verified adapter.
- Interrupted TRAINING (e.g. TIMEOUT): inspect actual complete `checkpoint-*`
  directories. Recovery requires adapter weights/config, trainer state, optimizer,
  scheduler and RNG state. The last checkpoint can be incomplete if writing was
  interrupted; choose a complete one. Preserve original logs/config/data.
- `Non-finite ... nan/inf`: numerical failure. Do not remove the guard, invent a
  completed score or switch settings for just that condition. Save the logs and
  bring the issue back for investigation. Other independent runs may continue.
- Input/output filesystem error: check job state, do not delete data or submit
  duplicates. Persistent errors need cluster support (`ccdsgpu-tc@ntu.edu.sg`).
- Missing model cache/offline error: the base models must be cached for the account
  running the job. Do not download large models inside GPU jobs or use someone
  else's directories without authorised shared access.

Recovery command TEMPLATE (replace style/size/seed and checkpoint-N correctly):

```bash
sbatch scripts/train/slurm_final.sh long 15B 1 --resume-from-checkpoint ckpt/long_15B_s1/checkpoint-N
```

This template cannot be copied unchanged: N must be an existing complete
checkpoint number. Do not use the command to resume a numerical failure blindly.
The full run retains its original frozen data/settings; resumption is not a new
seed or a new experiment. The save/resume integration test passed in job 70805.

## 8. Archive small evidence and back up weights

### On TC1, after runs finish

```bash
python -m reasoning.train.collect_records
```

Generate the list of evaluated adapters and SHA-256 checksums (paste this whole
Python block, including the final PY line, into TC1):

```bash
python - <<'PY'
import glob, hashlib, json
from pathlib import Path
records = [json.loads(Path(p).read_text()) for p in glob.glob('scripts/train/records/final_runs/*.json')]
records = {r['run_id']: r for r in records if r.get('status') == 'evaluated' and r.get('protocol_id') == 'sft-v1-20261031'}
out = Path('scripts/train/records')
paths, checksums = [], []
for name, record in sorted(records.items()):
    adapter = Path(record['adapter'])
    weights = adapter / 'adapter_model.safetensors'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != record['adapter_sha256']:
        raise ValueError(f'Weights changed after evaluation: {name}')
    paths.append(str(adapter) + '\n')
    checksums.append(f'{digest}  {adapter.name}/adapter_model.safetensors\n')
(out / 'final-adapter-paths.txt').write_text(''.join(paths))
(out / 'final-weights.sha256').write_text(''.join(checksums))
print(f'Listed {len(records)}/24 evaluated adapters for backup')
PY
```

Review and commit the SMALL evidence only:

```bash
git status
git add scripts/train/records
git commit -m "Archive completed final adapter evidence"
git push origin b/partb
```

Never `git add -f ckpt` or commit weights/raw generations. If authentication or
Git author configuration fails, the cluster files still exist; resolve Git
separately and do not rerun training to fix a Git problem.

### On the Mac LOCAL terminal (not inside TC1 SSH)

Open a second local Terminal window, or `exit` SSH. Stay connected to NTU/VPN.

```bash
mkdir -p ~/Downloads/sc4001-backups/adapters
scp -r jchua134@10.96.189.11:~/sc4001-group-project/scripts/train/records ~/Downloads/sc4001-backups/
```

Copy only the evaluated adapter folders from the generated list (paste the whole
loop). It may ask for your NTU password for each connection. The original copies
stay on TC1. Nested Trainer checkpoints are included and increase transfer size.

```bash
while IFS= read -r adapter <&3; do
  scp -r "jchua134@10.96.189.11:~/sc4001-group-project/$adapter" "$HOME/Downloads/sc4001-backups/adapters/" || break
done 3< "$HOME/Downloads/sc4001-backups/records/final-adapter-paths.txt"
```

Verify downloaded weights against the cluster checksums:

```bash
cd ~/Downloads/sc4001-backups/adapters
shasum -a 256 -c ../records/final-weights.sha256
```

Every listed adapter should print OK. Missing/FAILED files are not a verified
backup; retry the affected copy before marking it backed up. Do not delete the
cluster copies. TC1 explicitly provides NO home-directory backup.

Finally, in the Mac project checkout, `git pull --ff-only origin b/partb` brings
the small committed evidence to the Mac too. Confirm/distribute approved shared
paths to Person C separately; no message is sent by these commands.
