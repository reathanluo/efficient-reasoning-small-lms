# Teacher job scripts (owner: Person A)

`slurm_teacher.sh` runs one teacher step on TC1: everything after the script name
is passed to `python -m`, and `--dtype float16` is added for the vLLM steps. The
output format is in `reasoning/teacher/__init__.py`.

## How `data/sft/` was made (TC1 V100, 27 Oct 2026)

Run from the repo root on the head node. Each `sbatch` is one GPU job; TC1 gives
one GPU per user, so they run one after another. Raw output goes to
`data/sft/raw/` (gitignored).

```bash
S=scripts/teacher/slurm_teacher.sh

# 1. Standard rationales: Qwen2.5-7B-Instruct, 1 attempt, then 4 more for misses (jobs 70033, 70069)
sbatch -J t-std $S reasoning.teacher.generate --style standard --out data/sft/raw/standard.jsonl
sbatch -J t-std-retry $S reasoning.teacher.generate --style standard --n 4 \
    --only-missing data/sft/raw/standard.jsonl --out data/sft/raw/standard.retry1.jsonl

# 2. Long rationales: DeepSeek-R1-Distill-Qwen-7B in three parts under the 6 h limit,
#    then 4 more attempts for misses (jobs 70040, 70041, 70054, 70085)
sbatch -J t-long0 $S reasoning.teacher.generate --style long --start 0    --limit 2325 --out data/sft/raw/long.part0.jsonl
sbatch -J t-long1 $S reasoning.teacher.generate --style long --start 2325 --limit 2325 --out data/sft/raw/long.part1.jsonl
sbatch -J t-long2 $S reasoning.teacher.generate --style long --start 4650              --out data/sft/raw/long.part2.jsonl
sbatch -J t-long-retry $S reasoning.teacher.generate --style long --n 4 \
    --only-missing data/sft/raw/long.part{0,1,2}.jsonl --out data/sft/raw/long.retry1.jsonl

# 3. Rejection sampling (CPU only: run it on a laptop or inside a job, not on the head node)
python -m reasoning.teacher.filter --style standard data/sft/raw/standard.jsonl data/sft/raw/standard.retry1.jsonl \
    --out data/sft/raw/standard.filtered.jsonl
python -m reasoning.teacher.filter --style long data/sft/raw/long.part{0,1,2}.jsonl data/sft/raw/long.retry1.jsonl \
    --out data/sft/raw/long.filtered.jsonl

# 4. Concise: Qwen2.5-7B-Instruct rewrites each long rationale, 2 attempts, four parts (jobs 70112-70123)
for p in 0 1 2 3; do
  sbatch -J t-con$p $S reasoning.teacher.compress data/sft/raw/long.filtered.jsonl \
      --n 2 --start $((p*1700)) --limit 1700 --out data/sft/raw/concise.part$p.jsonl
done
cat data/sft/raw/concise.part{0,1,2,3}.jsonl > data/sft/raw/concise.filtered.jsonl

# 5. The four datasets on shared ids, plus stats.json (CPU only)
python -m reasoning.teacher.build --standard data/sft/raw/standard.filtered.jsonl \
    --long data/sft/raw/long.filtered.jsonl --concise data/sft/raw/concise.filtered.jsonl --out data/sft
```

Result: 6,973 problems → standard 6,797, long 6,738, concise 6,600 kept → 6,547
in every style. Mean rationale length in Qwen2.5 tokens: long 1,209, standard 242,
concise 37 (`data/sft/stats.json`). Sampling is seeded (seed 0), but vLLM batching
on a different GPU can still change individual samples.
