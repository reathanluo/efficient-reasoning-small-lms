# Results (owner: Person C)

Small, committed outputs that the report uses:

- `tables/`: `*.summary.json` files copied from `runs/`, plus CSV tables
- `scored/`: `*.scored.jsonl` files copied from `runs/` (per question: each sample's answer, correctness, generated tokens and finish reason, in sampling order). They let anyone rerun the analysis without a GPU, because TC1 does not let anyone read another user's `runs/`.
- `figures/`: final plots (PNG or PDF), named the way the report cites them

Raw generations stay in `runs/`, which is gitignored because it is too large.

Files directly in `scored/` and `tables/` are from `gsm8k_val`. Other splits go in a subfolder named after the split, because a run has the same file name on every split. For example, to publish the `gsm8k_test` runs:

```bash
mkdir -p results/scored/gsm8k_test results/tables/gsm8k_test
cp runs/baselines/gsm8k_test/*.scored.jsonl runs/students/gsm8k_test/*.scored.jsonl results/scored/gsm8k_test/
cp runs/baselines/gsm8k_test/*.summary.json runs/students/gsm8k_test/*.summary.json results/tables/gsm8k_test/
cp runs/baselines/gsm8k_test/*.meta.json runs/students/gsm8k_test/*.meta.json results/tables/gsm8k_test/
```

## What is here

### Baselines on `gsm8k_val`

These are the untrained models: TC1 job 70474, 29 Oct 2026, V100, float16, vLLM 0.7.3.

- `tables/baselines_gsm8k_val.csv` is one row per run, built with `python -m reasoning.eval.table`.
- `tables/*.summary.json` and `tables/*.meta.json` are the full summary and the exact settings of each run.
- `scored/*.scored.jsonl` holds per-question, per-sample answers, tokens and finish reasons.

Read these with three caveats:

- **Base + chat prompt.** This is the untrained starting point of every student. The 1.5B base model produces nonsense with the chat template: it repeats one token, or echoes the question until the 1,024-token cap. It does the same in float32 on a Mac, so this is the base model's behaviour, not float16 and not a harness bug. Its chat-token embeddings are small (norm 0.41, against about 1.0 for ordinary words), which fits a model that never saw the chat format. The 0.5B base copes (43.8%). Fine-tuning does teach this format: the 1.5B standard students reach about 86%.
- **Possible contamination.** `gsm8k_val` is 500 problems taken from the GSM8K *train* split, which Qwen2.5 may have seen in pretraining. Its numbers may be optimistic, so compare conditions with each other, not with published GSM8K scores. GSM-Symbolic is the contamination check.
- **Sampled single answers.** At T = 0.7, one answer (maj@1) is less accurate than greedy decoding. Voting overtakes greedy from about k = 4 to 8.

### Students on `gsm8k_val`

These are Person B's 24 adapters (protocol `sft-v1-20261031`): TC1 jobs 71218, 71221, 71250, 71712 and 71713 generated them and job 71783 scored them, 1-2 Nov 2026, V100, float16, vLLM 0.7.3, 4,096-token cap.

- `tables/students_gsm8k_val.csv` is one row per run (48 runs: greedy and 16 samples for each adapter), built with `python -m reasoning.eval.table`.
- `tables/students_mean_gsm8k_val.csv` holds the mean over the three training seeds at each k, built with `python -m reasoning.eval.plot`.
- `figures/accuracy_vs_tokens_gsm8k_val.png` and `.pdf` plot those means.

Read these with four caveats, plus the contamination caveat above:

- **Scoring fix.** These were scored with the `is_correct` overflow fix from PR #8. Re-scoring the 18 runs that had already finished gave identical output.
- **Runaway answers.** A few answers never stop and hit the 4,096-token cap, which pulls up mean token costs. Greedy concise has a mean of 101 tokens (1.5B) and 162 (0.5B), but a median of about 38. `scored/` has per-sample tokens for medians and budget analyses.
- **Long style.** About 30% of greedy long answers and 18-25% of sampled ones hit the cap. Person B's lower long scores used a 512-token cap, so don't compare them with these.
- **Folder names.** `long_15B_s0_pilot_70806` and `standard_05B_s0_pilot_70807` are seed 0 of their conditions, as recorded in Person B's handoff manifest.
