# Rationale Length or More Votes? (SC4001 Group 77, Topic G)

Code and results for our SC4001 Neural Networks and Deep Learning group project: Chen Han, Chua Jun Hong and Luo Yuxuan.

**Research question:** for Qwen2.5-0.5B and Qwen2.5-1.5B on GSM8K maths problems, how do rationale-training style and self-consistency affect accuracy under a common average inference-token budget?

We fine-tuned both base models with LoRA on the same 6,547 GSM8K training problems written in four styles (long, standard, concise and answer-only rationales from larger teacher models), with 3 seeds each: 24 adapters. We then evaluated every adapter on the 1,319 GSM8K test questions with greedy decoding and with self-consistency voting, and counted generated tokens per question as the cost. The hypotheses and the success criteria we fixed before testing are in [docs/research-question-card.md](docs/research-question-card.md).

**Main results on `gsm8k_test`** (mean over 3 seeds, generated tokens per question in brackets):

| Training style | 0.5B, greedy | 0.5B, best vote | 1.5B, greedy | 1.5B, best vote |
|---|---|---|---|---|
| Standard | 48.7 (315) | 61.1 (4,681) | 74.9 (277) | 83.5 (4,349) |
| Concise | 32.3 (199) | 38.8 (1,243) | 54.2 (115) | 61.2 (941) |
| Long | 31.9 (1,862) | 44.0 (16,541) | 59.8 (1,928) | 79.3 (14,825) |
| Answer-only | 10.6 (5.5) | 9.1 (85) | 13.4 (5.5) | 12.7 (85) |
| Instruct, no training | 46.9 (301) | 60.4 (4,652) | 73.5 (291) | 82.5 (4,353) |
| Base 8-shot | 36.6 (123) | 45.6 (1,672) | 63.2 (101) | 73.7 (1,582) |

Best vote is maj@16, or maj@8 for long. In short:

- **H1 is not supported.** Within 512 tokens, concise students with voting lose to one greedy answer from the standard student, by 17.4 points at 0.5B and 16.8 points at 1.5B.
- **H2 is not supported.** Concise answers are more varied, not less, and their voting gain is smaller than standard's at 0.5B but larger at 1.5B.
- **H3 holds only in its truncation part.** About a third of greedy long answers hit the 4,096-token cap, but at 1.5B long students are 5.6 points more accurate than concise ones.

## Team

| Role | Name | GitHub | Work |
|---|---|---|---|
| Person A: teacher data | Chen Han | [@Chen-Han-TX](https://github.com/Chen-Han-TX) | `reasoning/teacher/`, `scripts/teacher/`, `data/sft/` |
| Person B: training | Chua Jun Hong | [@leclerc162003](https://github.com/leclerc162003) | `reasoning/train/`, `scripts/train/`, `configs/` |
| Person C: evaluation | Luo Yuxuan | [@reathanluo](https://github.com/reathanluo) | `reasoning/eval/`, `scripts/eval/`, `results/` |

The shared code (`reasoning/data.py`, `prompts.py`, `answer.py`, `stats.py`, `data/splits/`, `tests/`) was changed only through pull requests reviewed by the other two members. The git workflow is in [CONTRIBUTING.md](CONTRIBUTING.md).

## What each folder holds

| Folder | What it is | Owner | In git? |
|---|---|---|---|
| `reasoning/` | The project's Python package. Run its modules from the repo root with `python -m reasoning.<module>`. | — | yes |
| `reasoning/data.py`, `prompts.py`, `answer.py`, `stats.py` | Shared core: the data splits, the one prompt format, the one answer parser, statistics helpers | Shared | yes |
| `reasoning/teacher/` | Teacher generation (`generate.py`), rejection sampling (`filter.py`), concise rewriting (`compress.py`) and building the four datasets (`build.py`). `__init__.py` holds the output format. | A | yes |
| `reasoning/train/` | LoRA fine-tuning (`sft.py`, `run_final.py`), data and record checks, and their tests. `__init__.py` holds the adapter format. | B | yes |
| `reasoning/eval/` | `generate.py` (run a model), `evaluate.py` (score a run), `table.py` (CSV tables), `plot.py` (Figure 1), `hypotheses.py` (paired tests of H1-H3), `compare.py` (any two runs) | C | yes |
| `scripts/teacher/`, `scripts/train/`, `scripts/eval/` | SLURM job scripts for the TC1 cluster, each with a README of the exact jobs we ran | A / B / C | yes |
| `configs/` | The frozen training settings (`sft-final.json`) and the tuning results (`tuning-results.csv`) | B | yes |
| `data/splits/` | Fixed train / val / test / out-of-distribution splits. Everyone used these exact files. | Shared | yes |
| `data/sft/` | The four training datasets (`long`, `standard`, `concise`, `answer_only`) and `stats.json` | A | yes; `raw/` is not |
| `results/` | Per-question scored answers, run summaries, CSV tables, figures and the paired tests. See [results/README.md](results/README.md). | C | yes |
| `docs/` | The research question card, the assignment handout and FAQ, and a snapshot of our planning handbook | Everyone | yes |
| `tests/` | Unit tests for the shared code and evaluation. GitHub runs them on every push. | Shared | yes |
| `runs/` | Raw model outputs from `reasoning.eval.generate` (large) | C | **no**, stays on the cluster |
| `ckpt/` | Trained LoRA adapters (large) | B | **no**, see step 3 below |

## Reproducing the results

The pipeline has five steps. Steps 2 to 4 need a GPU; we ran them on NTU's TC1 cluster (one NVIDIA V100 32 GB per job, 6-hour limit). Every step's output is committed, so you can start at any step. **Step 5 rebuilds every table, the figure and the paired tests from the committed files in a few seconds, with no GPU.**

| Step | What | GPU time we used | Output (committed) |
|---|---|---|---|
| 1 | Data splits | none | `data/splits/` |
| 2 | Teacher rationales and the four training sets | about 6 h | `data/sft/` |
| 3 | 24 LoRA adapters | about 20 h | GitHub release `person-b-adapters-20261101`, records in `scripts/train/records/` |
| 4 | Evaluation on `gsm8k_test` | about 32 h | `results/scored/gsm8k_test/`, `results/tables/gsm8k_test/` |
| 5 | Tables, figure and paired tests | none | `results/tables/`, `results/figures/` |

For dependencies, set up either the cluster (GPU steps) or a laptop (step 5 and the tests), as described in the sections further down.

### Step 1: data splits

```bash
python -m reasoning.data --out data/splits
```

This downloads GSM8K, SVAMP and GSM-Symbolic and writes the splits in the table under [Data preparation](#data-preparation). The 500 validation problems are held out from GSM8K train with seed 0. All tuning used `gsm8k_val`; `gsm8k_test` was used only for the final runs.

### Step 2: teacher data (Person A)

The exact jobs, in order, are in [scripts/teacher/README.md](scripts/teacher/README.md). In short, on TC1:

```bash
S=scripts/teacher/slurm_teacher.sh
sbatch $S reasoning.teacher.generate --style standard --out data/sft/raw/standard.jsonl   # Qwen2.5-7B-Instruct
sbatch $S reasoning.teacher.generate --style long --out data/sft/raw/long.jsonl           # DeepSeek-R1-Distill-Qwen-7B (we split it into 3 jobs)
python -m reasoning.teacher.filter --style standard data/sft/raw/standard.jsonl --out data/sft/raw/standard.filtered.jsonl   # keep correct answers
python -m reasoning.teacher.filter --style long data/sft/raw/long.jsonl --out data/sft/raw/long.filtered.jsonl
sbatch $S reasoning.teacher.compress data/sft/raw/long.filtered.jsonl --n 2 --out data/sft/raw/concise.filtered.jsonl
python -m reasoning.teacher.build --standard data/sft/raw/standard.filtered.jsonl \
    --long data/sft/raw/long.filtered.jsonl --concise data/sft/raw/concise.filtered.jsonl --out data/sft
```

The result is `data/sft/{long,standard,concise,answer_only}.jsonl` (6,547 shared problems each) and `data/sft/stats.json`, the source of the report's Table 1.

### Step 3: LoRA training (Person B)

The frozen settings are in `configs/sft-final.json` (protocol `sft-v1-20261031`: rank 8, alpha 16, learning rate 1e-4, one epoch, FP32, 4,096-token limit). One job trains one style, size and seed:

```bash
sbatch scripts/train/slurm_final.sh standard 15B 0     # styles: long standard concise answer_only; sizes: 05B 15B; seeds: 0 1 2
python -m reasoning.train.run_final standard 15B 0 --dry-run   # preview an entry without a GPU
```

Each run writes `ckpt/<style>_<size>_s<seed>/`. Two seed-0 adapters kept their pilot names, `ckpt/long_15B_s0_pilot_70806` and `ckpt/standard_05B_s0_pilot_70807`. The full procedure, including the pilots and recovery, is in [scripts/train/final-runs.md](scripts/train/final-runs.md). The hyperparameter sweep behind the report's Table 2 is in `configs/tuning-results.csv`.

To skip training, download our 24 adapters (611 MB) from the GitHub release `person-b-adapters-20261101` of this repository and unpack them in the repo root. Access needs repository permission because the repository is private:

```bash
gh release download person-b-adapters-20261101 -R reathanluo/sc4001-group-project
tar -xzf sc4001-final-backup-20261101.tar.gz     # creates ckpt/ (plus copies of configs/ and the training records)
cd ckpt && sha256sum -c ../scripts/train/records/final-weights.sha256 && cd ..   # expect 24 OK (shasum -a 256 -c on a Mac)
```

### Step 4: evaluation on `gsm8k_test` (Person C)

The ten jobs, with the exact commands, are in [scripts/eval/README.md](scripts/eval/README.md). For example:

```bash
sbatch scripts/eval/slurm_baselines.sh gsm8k_test                                   # untrained baselines (Instruct, base chat, base 8-shot)
sbatch scripts/eval/slurm_students.sh gsm8k_test ckpt/standard_15B_*                # greedy + 16 samples per adapter
RUNS="greedy sc8" sbatch --export=ALL scripts/eval/slurm_students.sh gsm8k_test ckpt/long_15B_s1   # long: 8 samples
```

Students may write up to 4,096 tokens and baselines up to 1,024. Samples use temperature 0.7. Long students used 8 samples instead of 16 on the test set, because 16 do not fit in one 6-hour job. Each job writes raw generations to `runs/` and scores them; we then copied the scored files into `results/` as described in [results/README.md](results/README.md).

To run and score one model by hand:

```bash
python -m reasoning.eval.generate --model Qwen/Qwen2.5-1.5B --lora ckpt/standard_15B_s1 --prompt chat \
    --split gsm8k_test --n 16 --temperature 0.7 --max-new-tokens 4096 --out runs/standard_15B_s1_sc16.jsonl
python -m reasoning.eval.evaluate runs/standard_15B_s1_sc16.jsonl
```

### Step 5: tables, figure and paired tests (no GPU)

Run from the repo root. Each command rebuilds a committed file. The bootstrap is seeded, but a p-value can differ in the third decimal between Python versions.

```bash
T=results/tables/gsm8k_test
# One row per run (report Table 3, Figure 2): baselines, then students
python -m reasoning.eval.table $T/base*.summary.json $T/inst*.summary.json --out results/tables/baselines_gsm8k_test.csv
python -m reasoning.eval.table $(ls $T/*.summary.json | grep -vE '/(base|inst)') --out results/tables/students_gsm8k_test.csv
# Seed means and Figure 1 (needs matplotlib)
python -m reasoning.eval.plot $T --split gsm8k_test \
    --out results/figures/accuracy_vs_tokens_gsm8k_test --csv results/tables/students_mean_gsm8k_test.csv
# Paired tests of H1-H3 (report Table 4 and the H3 section)
python -m reasoning.eval.hypotheses results/scored/gsm8k_test > results/tables/hypotheses_gsm8k_test.txt
```

Where the numbers in the report come from:

| In the report | File |
|---|---|
| Table 1 (training sets) | `data/sft/stats.json` |
| Table 2 (LoRA rank) | `configs/tuning-results.csv` |
| Table 3 and Figure 1 | `results/tables/students_mean_gsm8k_test.csv`, `results/tables/baselines_gsm8k_test.csv`, `results/figures/accuracy_vs_tokens_gsm8k_test.png` |
| Table 4 and the H3 tests | `results/tables/hypotheses_gsm8k_test.txt` |
| Figure 2 (truncation and `####` format) | the `truncation_rate` and `format_hash` columns of `results/tables/students_gsm8k_test.csv`, averaged over seeds |
| Training time | `scripts/train/methods-draft.md` (recorded per-run times) |
| Table A1 (hyperparameters) | `configs/sft-final.json` |

`python -m reasoning.eval.compare A.scored.jsonl B.scored.jsonl --k 1` compares any two scored runs with a paired bootstrap and an exact McNemar test.

## Setup on the TC1 cluster (NTU CCDS GPU cluster)

From the TC1 user guide (Feb 2026). TC1 is reachable only from the NTU network: off campus, connect the NTU GlobalProtect VPN first (https://vpngate.ntu.edu.sg).

| Fact | Value |
|---|---|
| Head node | `10.96.189.11` (`CCDS-TC1`), login with your NTU network ID in lower case |
| GPUs | NVIDIA V100 32 GB: no bfloat16, so use `--dtype float16` / `DTYPE=float16` |
| Partition, QoS | `UGGPU-TC1`, `normal` (check yours with `MyTCinfo`) |
| QoS `normal` limits | 1 GPU, 20 CPUs, 64 GB memory, 2 jobs at a time, 6 hours per job |
| Disk | About 100 GB home quota. Check with `ncdu`; nothing is backed up. |

**Rules:** never run code on the head node (only setup commands such as `conda` and `pip`); run everything through `sbatch` job scripts, not `srun`; use only the QoS assigned to you. Breaking these gets jobs killed without notice.

**One-time setup** (in an SSH session on the head node):

```bash
ssh <ntu-id>@10.96.189.11
git clone git@github.com:reathanluo/sc4001-group-project.git && cd sc4001-group-project   # or copy the submitted ZIP over and unzip it
module load uv
uv venv --seed --python 3.11 .venv       # the job scripts use .venv/ when it exists
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/uv-cache-$USER UV_LINK_MODE=copy   # home storage has no file locks
uv pip install -r requirements.txt
python -m pytest tests                   # must print 31 passed
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"   # 0.7.3 2.5.1

# Download the models once on the head node; jobs read them offline.
export HF_HOME=$HOME/hf_cache
for m in Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct \
         Qwen/Qwen2.5-7B-Instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; do   # last two: teachers (A)
  huggingface-cli download $m            # newer huggingface_hub: hf download $m
done
```

Use uv, not conda: on TC1 (27–29 Oct), `conda create` fails while unpacking `ncurses` with `Too many levels of symbolic links`. After that, `pip install` silently falls back to the shared Python 3.14, which vLLM 0.7.3's `numba` refuses. Run `source .venv/bin/activate` in every new session before using `python`. Cloning over SSH needs TC1's own key (`cat ~/.ssh/id_ecdsa.pub`) added to your GitHub account.

**Run a job:**

```bash
sbatch scripts/eval/slurm_baselines.sh   # prints a job id
squeue -u <ntu-id>                       # PD = waiting, R = running
tail -f slurm-g-baselines-<jobid>.out    # follow the log
scancel <jobid>                          # stop it
seff <jobid>                             # after it ends: CPU and memory actually used
```

Each job writes `slurm-<job-name>-<jobid>.out` and `.err` in the repo root (gitignored). If a job stays `PD` with a `QOS...Limit` reason, it asks for more than the QoS allows: lower `--mem`, `--cpus-per-task` or `--time`.

**Checked on a TC1 V100 (27 Oct, jobs 70022–70033):**

- `requirements.txt` pins vLLM 0.7.3 (torch 2.5.1). The latest wheels (vLLM 0.32, torch 2.13) ship no V100 (sm_70) kernels and fail on TC1. If you installed before 27 Oct, rerun `pip install -r requirements.txt`.
- Prefix caching is off by default in `reasoning/eval/generate.py`: on V100 a cache hit aborts with `mma -> mma layout conversion is only supported on Ampere`. Small runs can pass and the full run then crashes, so this is not optional.
- The QoS allows 2 queued jobs but only 1 GPU, so a second GPU job waits with reason `QOSMaxGRESPerUser` until the first ends.

## Local setup (laptop, no GPU)

Enough for step 5 above, the tests and debugging on a few questions. vLLM needs Linux + CUDA, so it is left out here; use the cluster setup above for real runs. [uv](https://docs.astral.sh/uv/) downloads Python 3.11 for you, and `.venv/` is gitignored.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install transformers datasets pytest torch matplotlib
python -m pytest tests          # should print 31 passed
python -m pytest reasoning/train   # training checks: 15 passed (3 skip if torch is missing)
```

In VS Code, pick the `.venv` interpreter (Cmd+Shift+P → Python: Select Interpreter). Run `source .venv/bin/activate` in every new terminal.

## Data preparation

The splits are committed in `data/splits/`, so everyone uses identical data. To regenerate them:

```bash
python -m reasoning.data --out data/splits
```

| split | size | use |
|---|---|---|
| `gsm8k_train` | 6,973 | exemplars, teacher generation, SFT |
| `gsm8k_val` | 500 | **all tuning and debugging** (held out from GSM8K train, seed 0) |
| `gsm8k_test` | 1,319 | **final runs only** |
| `svamp` | 1,000 | out-of-distribution eval |
| `gsm_symbolic` | 5,000 | perturbed GSM8K-test templates; robustness and contamination check |
| `exemplars` | 8 | fixed few-shot exemplars (from `gsm8k_train`) |

## Rules we followed

- **Never tune on `gsm8k_test`.** Prompts, hyperparameters, checkpoints and decoding settings are chosen on `gsm8k_val`. `generate.py` warns when the test split is used.
- **One prompt, one target format.** SFT inputs come from `prompts.chat_prompt(tokenizer, question)` and targets from `prompts.format_target(rationale, gold)`. Only the rationale differs between conditions.
- **One parser.** Every answer is scored by `reasoning/answer.py`.
- **Keep raw generations** in `runs/`. Scoring is a separate step, so a parser fix never needs a GPU rerun.
- **Report cost three ways:**
  - generated tokens per question (primary);
  - approximate FLOPs, 2·N·(prompt + generated);
  - wall-clock time.
  
  The 8-shot prompt's prefix is the same for every question, so report its cost separately. On TC1 prefix caching is off (see above), so wall-clock time includes recomputing it.
- **Self-consistency:** 16 samples at T = 0.7 (8 for long students on `gsm8k_test`). maj@k is the mean over all subsets of k samples, or over 128 random subsets when there are too many.

## Acknowledgements

We wrote the code in `reasoning/`, `scripts/` and `tests/` ourselves, with help from AI tools as described in the report's AI Use Statement. It builds on these existing models, datasets and libraries:

- **Models:** Qwen2.5-0.5B, Qwen2.5-1.5B and their Instruct versions (students and baselines), Qwen2.5-7B-Instruct (standard and concise teacher) and DeepSeek-R1-Distill-Qwen-7B (long teacher), all from Hugging Face.
- **Datasets:** GSM8K (`openai/gsm8k`), SVAMP (`ChilleD/SVAMP`) and GSM-Symbolic (`apple/GSM-Symbolic`), from Hugging Face. SVAMP and GSM-Symbolic were prepared but, in the end, not evaluated.
- **Libraries:** vLLM for generation, Hugging Face Transformers, Datasets, PEFT and Accelerate for training, PyTorch, matplotlib and pytest. Versions are pinned in `requirements.txt`; the exact TC1 environment is in `configs/tc1-smoke-packages.txt`.
- **Methods:** LoRA (Hu et al., 2022) and self-consistency (Wang et al., 2023). The full reference list is in the report.
