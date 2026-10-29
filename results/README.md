# Results (owner: Person C)

Small, committed outputs that the report uses:

- `tables/`: `*.summary.json` files copied from `runs/`, plus CSV tables
- `scored/`: `*.scored.jsonl` files copied from `runs/` (per question: each sample's answer, correctness, generated tokens and finish reason, in sampling order). Person A's adaptive-voting analysis (H4) reads these, because TC1 does not let anyone read another user's `runs/`.
- `figures/`: final plots (PNG or PDF), named the way the report cites them

Raw generations stay in `runs/`, which is gitignored because it is too large.

To publish a finished batch of runs:

```bash
mkdir -p results/scored results/tables
cp runs/baselines/gsm8k_val/*.scored.jsonl results/scored/
cp runs/baselines/gsm8k_val/*.summary.json results/tables/
```
