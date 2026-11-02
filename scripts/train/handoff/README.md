# Person B adapter handoff, 1 November 2026

All 24 adapters have completed training and the common validation check. The
restored Mac backup was independently checked: weights/config/data match the
frozen protocol, all 24 weight SHA-256 hashes match, and all 12,000 per-question
correctness flags reproduce with the shared parser. The handoff destination and
receipt by Person C are still pending.

## Files for Person C

- `configs/sft-final.json`: frozen protocol `sft-v1-20261031`.
- `scripts/train/records/final-handoff-manifest.csv`: all 24 actual adapter paths,
  base models, seeds, data/weight hashes, train/eval job IDs and score paths.
- `scripts/train/records/final-validation-summary.csv`: eight condition summaries.
- `scripts/train/records/final_runs/`: original final-run registration evidence.
- `scripts/train/records/final-backup-audit.json`: Mac restore/score audit.
- Verified archive: `sc4001-final-backup-20261101.tar.gz` (611,286,531 bytes).
- SHA-256: `ab5d366494eee8dd911b4fcb04a72ce21c37ada1a36a6fb8947bb0daead0f6dd`.

The archive contains weights, adapter configs, training logs, all 24 converted
per-epoch vLLM prediction files, the frozen configuration and collected small
records. It deliberately excludes optimizer checkpoints, base-model weights and
the ORIGINAL raw `runs/final_validation` folder. Original raw generations remain
on TC1; a separate backup command is provided below. The converted prediction
files retain IDs/gold/text/correctness/token counts/finish reasons, but are not
the evaluator's original raw JSONL schema with per-question prompt-token counts.

## Extract and verify

Extract the archive into a project/work directory while preserving its relative
`ckpt/`, `configs/` and `scripts/train/records/` folders. It contains PEFT adapters,
not standalone merged models. Download/cache the matching base models separately.
Inside the restored `ckpt` directory:

```bash
shasum -a 256 -c ../scripts/train/records/final-weights.sha256
```

On Linux, use `sha256sum -c` instead. Expect 24 OK entries. Two actual adapter
folders retain pilot names; use the manifest rather than constructing all paths:

- `long_15B_s0` -> `ckpt/long_15B_s0_pilot_70806`
- `standard_05B_s0` -> `ckpt/standard_05B_s0_pilot_70807`

## Evaluation contract

Students are Qwen2.5-0.5B / Qwen2.5-1.5B BASE models, not Instruct versions.
Use the shared chat prompt and shared answer parser. Training used FP32; validated
vLLM inference uses FP16 on TC1 V100. Prefix caching remains disabled. Environment
versions are recorded in `configs/tc1-smoke-packages.txt`; its machine-local
packaging entry means it is provenance rather than a portable install file.

Common completed monitoring: all 500 GSM8K validation rows, greedy decoding,
one sample, seed 0, max 512 new tokens, max model length 8192. The 24 adapters use
training seeds 0/1/2. Monitoring seeds and training seeds serve different purposes.
No final test or OOD evaluation was run as part of this handoff.

Training setup: one epoch, LR 1e-4, LoRA rank/alpha 8/16, dropout 0.05, batch 1,
accumulation 8, max training sequence 4096, completion-only loss and EOS.
Keep settings frozen; do not retune individual low-scoring conditions.

Long-style scores are dominated by the monitoring cap: across three seeds,
truncation is 92.4% for 0.5B and 93.0% for 1.5B. The seed-0 1.5B diagnostic at
caps 512/2048/4096 scored 28.2%/64.6%/62.6% (different validation attempts; retain
them all). Do not replace a common-budget score with a condition's best-budget
score. C's planned token-budget/self-consistency/test/OOD evaluations remain to do.

## Github delivery

Commit small records and documentation to `b/partb`; merge through the repository's
normal PR workflow. Do not commit weights or this large archive into Git history.
The archive can be attached as an asset to a GitHub Release instead. GitHub's
documented per-asset limit is under 2 GiB; this archive fits. Repository readers
can access published releases; release managers need write permission.

Official reference: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

Prepared release title: `Person B: 24 verified LoRA adapters (1 Nov 2026)`.
Suggested tag: `person-b-adapters-20261101`, targeted at the reviewed/pushed
`b/partb` commit. This is a suggested new tag, not an existing published release.
Use `release-notes.md` as the description. Attach the 583 MiB archive, its checksum
sidecar, the handoff manifest and optionally the separate raw-validation archive.
GitHub's automatic Source code ZIP/TAR does not contain the adapter weights.
Publishing a release may notify repository watchers; no release or notification
has been created by Codex. Check the repository visibility before publishing.

## Preserve original raw generations (no GPU work)

On TC1, from the project root, create an additional archive:

```bash
tar -czf "$HOME/sc4001-final-validation-20261101.tar.gz" runs/final_validation
sha256sum "$HOME/sc4001-final-validation-20261101.tar.gz"
```

On the Mac LOCAL terminal while connected to NTU/VPN:

```bash
scp jchua134@10.96.189.11:~/sc4001-final-validation-20261101.tar.gz ~/Downloads/
shasum -a 256 ~/Downloads/sc4001-final-validation-20261101.tar.gz
```

Compare the two hashes before marking that backup verified or uploading it.
The current Mac weight-backup audit does not claim this additional archive exists.

## Draft message to C (not sent)

All 24 LoRA adapters are trained, validation-scored, backed up and checksum-verified.
The common configuration is `sft-v1-20261031` (FP32 training, rank 8, LR 1e-4,
one epoch, 4096-token training limit, seeds 0/1/2). The manifest lists actual paths
and matching base models, including two pilot-named seed-0 folders. I'll provide
the verified archive through our GitHub Release [insert published link]. Please
verify the checksum after downloading. The current scores use the same 500-row
validation split and 512-token greedy cap, so long models are heavily truncated;
these are not final test results. The full token-budget, SC and OOD evaluations
remain with C. Please confirm that you can load the adapters before we mark the
handoff received.
