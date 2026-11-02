All 24 unmerged PEFT LoRA adapters for the four-style, two-size, three-seed matrix
have completed training and common validation. The Mac backup and every adapter
weight hash were verified on 1 November 2026. Scores were independently reproduced
from all 12,000 stored prediction texts with the shared answer parser.

Frozen protocol: `sft-v1-20261031`, in `configs/sft-final.json`.

- Base models: Qwen2.5-0.5B and Qwen2.5-1.5B (base versions).
- Training: FP32, one epoch, LR 1e-4, LoRA rank/alpha 8/16, dropout 0.05.
- Batch size 1, accumulation 8, max training sequence 4096, seeds 0/1/2.
- Each style uses the same ordered 6,547 GSM8K training questions.
- Monitoring: 500 validation questions, greedy FP16 vLLM, 512 new tokens.

The archive contains adapter weights/configs, training logs, converted prediction
snapshots, frozen config and small records. It excludes base-model weights,
optimizer checkpoints and the original raw `runs/final_validation` folder.
Use the separate raw-validation asset if attached; otherwise obtain that folder
from the owner through an approved shared location. The automatic GitHub source
archives do NOT include adapter weights.

Adapter archive: `sc4001-final-backup-20261101.tar.gz` (611,286,531 bytes).

SHA-256:

```text
ab5d366494eee8dd911b4fcb04a72ce21c37ada1a36a6fb8947bb0daead0f6dd
```

After extraction, run `sha256sum -c ../scripts/train/records/final-weights.sha256`
from the restored `ckpt` directory (macOS: `shasum -a 256 -c ...`). Expect 24 OKs.

Use `scripts/train/records/final-handoff-manifest.csv` for exact paths. Exceptions:

- `long_15B_s0`: `ckpt/long_15B_s0_pilot_70806`
- `standard_05B_s0`: `ckpt/standard_05B_s0_pilot_70807`

| Style | 0.5B mean validation accuracy | 1.5B mean validation accuracy |
|---|---:|---:|
| Standard | 59.1% | 84.9% |
| Concise | 40.5% | 60.5% |
| Answer-only | 10.8% | 19.9% |
| Long | 19.8% | 28.4% |

Means are across three seeds at the same 512-token cap. These are validation
results, not final test/OOD results or significance claims. Long-style outputs
truncate on about 92-93% of questions at this cap; the longer-budget diagnostic
is retained separately. Compare actual generated-token cost and matched budgets
before claiming one style is more efficient. Person C's full evaluation remains.
