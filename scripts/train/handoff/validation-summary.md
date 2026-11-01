# Completed validation matrix

Verified from restored run records on 1 November 2026. All conditions use 500
GSM8K validation questions, three training seeds (0/1/2), greedy vLLM inference,
one sample, a 512-new-token cap and max model length 8192. No test/OOD results
are represented here. The 24 adapters have passed training/config/hash checks;
all 12,000 recorded correctness flags reproduce using the shared answer parser.

| Model | Style | Seed 0 | Seed 1 | Seed 2 | Mean | Seed SD (percentage points) | Mean generated tokens | Mean truncation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | Standard | 60.2% | 59.2% | 58.0% | 59.1% | 1.10 | 245.7 | 2.5% |
| 0.5B | Concise | 40.2% | 41.2% | 40.2% | 40.5% | 0.58 | 56.5 | 2.8% |
| 0.5B | Answer-only | 10.4% | 10.2% | 11.8% | 10.8% | 0.87 | 5.5 | 0.0% |
| 0.5B | Long | 22.6% | 17.6% | 19.2% | 19.8% | 2.55 | 508.4 | 92.4% |
| 1.5B | Standard | 85.2% | 85.0% | 84.4% | 84.9% | 0.42 | 247.6 | 2.1% |
| 1.5B | Concise | 60.2% | 61.4% | 59.8% | 60.5% | 0.83 | 50.2 | 1.6% |
| 1.5B | Answer-only | 20.0% | 19.8% | 19.8% | 19.9% | 0.12 | 5.4 | 0.0% |
| 1.5B | Long | 28.4% | 32.0% | 24.8% | 28.4% | 3.60 | 508.8 | 93.0% |

Mean tokens and truncation are averaged over the three equal-size seed runs.
SD is the sample SD across seeds, not a per-question CI or a significance test.
Question-level comparisons require paired analysis because the questions repeat
across conditions/seeds.

At this common cap, standard rationales yield the highest accuracy, while concise
outputs use far fewer tokens. Neither observation by itself establishes Pareto
dominance at equal actual generated-token cost. Long conditions are predominantly
cap-limited. The separate 1.5B seed-zero budget diagnostic scored 28.2%, 64.6% and
62.6% at caps 512, 2048 and 4096, respectively; keep those attempts distinct from
the registered final cap-512 score of 28.4%.

Next analyses belong to the common evaluation protocol: budget curves, greedy
versus self-consistency, per-question paired tests, failure/truncation inspection,
then the agreed final test/OOD runs. Do not select different best budgets per
condition and compare them as though compute were matched.

Sources: `scripts/train/records/final_runs/*.json`, matching
`scripts/train/records/evaluation/final_validation/*/*.summary.json`, restored
`ckpt/*/validation_epoch_1_vllm.jsonl`, and `final-backup-audit.json`.
