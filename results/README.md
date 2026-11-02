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

## What is here

### Baselines on `gsm8k_val`

These are the untrained models: TC1 job 70474, 29 Oct 2026, V100, float16, vLLM 0.7.3.

- `tables/baselines_gsm8k_val.csv` is one row per run, built with `python -m reasoning.eval.table`.
- `tables/*.summary.json` and `tables/*.meta.json` are the full summary and the exact settings of each run.
- `scored/*.scored.jsonl` holds per-question, per-sample answers, tokens and finish reasons. The H4 replay reads these.

Read these with three caveats:

- **Base + chat prompt.** This is the untrained starting point of every student. The 1.5B base model produces nonsense with the chat template: it repeats one token, or echoes the question until the 1,024-token cap. It does the same in float32 on a Mac, so this is the base model's behaviour, not float16 and not a harness bug. Its chat-token embeddings are small (norm 0.41, against about 1.0 for ordinary words), which fits a model that never saw the chat format. The 0.5B base copes (43.8%). Fine-tuning has to teach this format, so the 1.5B students need an early check.
- **Possible contamination.** `gsm8k_val` is 500 problems taken from the GSM8K *train* split, which Qwen2.5 may have seen in pretraining. Its numbers may be optimistic, so compare conditions with each other, not with published GSM8K scores. GSM-Symbolic is the contamination check.
- **Sampled single answers.** At T = 0.7, one answer (maj@1) is less accurate than greedy decoding. Voting overtakes greedy from about k = 4 to 8.
