# Final runs on `gsm8k_test` (owner: Person C)

**Done, 2-4 Nov 2026.** All 10 jobs completed without hitting the time limit, and the results are in `results/` (see `results/README.md`). This page records how they were run.

Everything runs from one TC1 account, one job at a time. The GPU hours are estimates scaled from the `gsm8k_val` runs: the test set has 2.6 times as many questions.

**Settings are frozen.** Nothing changes after anyone has seen a test score. Prompts, caps, the 4,096-token limit and T = 0.7 all stay as they were on `gsm8k_val`.

**Long uses 8 votes, the other styles use 16.** On `gsm8k_test`, 16 long samples would take about 6 to 6.4 hours per adapter. A run only saves when it ends, so it could never finish inside the 6-hour job limit. Eight samples take about 3 hours and still cover k = 8, the value H2 is tested at. This was decided before any test run, because of the compute limit, and the report must say so. Voting comparisons with long stop at k = 8.

## Before the first job

Unpack Person B's adapters into `ckpt/` (`final-handoff-manifest.csv` lists them), and start from an up-to-date checkout:

```bash
git checkout main && git pull
```

## The jobs, in order

The QoS allows 2 jobs per user, so keep at most 2 submitted and add the next when one finishes.

| # | What | GPU hours | Command |
|---:|---|---:|---|
| 1 | Baselines | 1.8 | `sbatch scripts/eval/slurm_baselines.sh gsm8k_test` |
| 2 | Concise and answer-only | 2.8 | `sbatch scripts/eval/slurm_students.sh gsm8k_test ckpt/concise_* ckpt/answer_only_*` |
| 3 | Standard 0.5B | 2.8 | `sbatch scripts/eval/slurm_students.sh gsm8k_test ckpt/standard_05B_*` |
| 4 | Standard 1.5B | 3.2 | `sbatch scripts/eval/slurm_students.sh gsm8k_test ckpt/standard_15B_*` |
| 5 | Long 0.5B seed 0 | 3.4 | `RUNS="greedy sc8" sbatch --export=ALL scripts/eval/slurm_students.sh gsm8k_test ckpt/long_05B_s0` |
| 6 | Long 0.5B seed 1 | 3.4 | `RUNS="greedy sc8" sbatch --export=ALL scripts/eval/slurm_students.sh gsm8k_test ckpt/long_05B_s1` |
| 7 | Long 0.5B seed 2 | 3.4 | `RUNS="greedy sc8" sbatch --export=ALL scripts/eval/slurm_students.sh gsm8k_test ckpt/long_05B_s2` |
| 8 | Long 1.5B seed 0 | 3.7 | `RUNS="greedy sc8" sbatch --export=ALL scripts/eval/slurm_students.sh gsm8k_test ckpt/long_15B_s0_pilot_70806` |
| 9 | Long 1.5B seed 1 | 3.7 | `RUNS="greedy sc8" sbatch --export=ALL scripts/eval/slurm_students.sh gsm8k_test ckpt/long_15B_s1` |
| 10 | Long 1.5B seed 2 | 3.7 | `RUNS="greedy sc8" sbatch --export=ALL scripts/eval/slurm_students.sh gsm8k_test ckpt/long_15B_s2` |

That's about 32 GPU hours. Each job scores every finished run in its folder when it ends. If a job hits the time limit, submit the same command again.

## When they finish

Check that `runs/students/gsm8k_test/` has 48 `*.meta.json` files: 24 adapters x 2 runs each. Then publish them as `results/README.md` describes, and build the tables and figure with `python -m reasoning.eval.table` and `python -m reasoning.eval.plot`, pointing them at `results/tables/gsm8k_test/`.

## If time allows

These are the first thing to cut. Both use greedy decoding only:

```bash
RUNS=greedy sbatch --export=ALL scripts/eval/slurm_students.sh svamp
```
```bash
RUNS=greedy LIMIT=1000 sbatch --export=ALL scripts/eval/slurm_students.sh gsm_symbolic
```

Each should take about 3 GPU hours. `gsm_symbolic` is sorted by template with 50 questions each, so `LIMIT=1000` covers the first 20 templates rather than a random sample. The report must say so.
