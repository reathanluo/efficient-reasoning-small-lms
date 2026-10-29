# SC4001 Group Project: Efficient Reasoning with Small LMs (Topic G)

**Research question:** for Qwen2.5-0.5B/1.5B on grade-school maths (GSM8K), is a fixed token budget better spent:

- in training, by LoRA-distilling long, standard or concise rationales from a teacher, or
- at inference, by self-consistency (majority vote over several samples)?

And do the two compose? The hypotheses and success criteria are in [docs/research-question-card.md](docs/research-question-card.md).

**Deadline:** Mon 16 Nov 2026, 11:59 PM. We aim to submit on Sun 15 Nov.

## What each person does

Each person owns one stage of the pipeline: its folders, its jobs on the cluster, and the report sections that describe it.

| Role | Name | GitHub |
|---|---|---|
| Person A: teacher data | Chen Han | [@Chen-Han-TX](https://github.com/Chen-Han-TX) |
| Person B: training | Chua Jun Hong | [@leclerc162003](https://github.com/leclerc162003) |
| Person C: evaluation | Luo Yuxuan | [@reathanluo](https://github.com/reathanluo) |

Everyone can read and run anything. To change a folder you don't own, open a pull request (PR) and tag its owner. The git workflow is in [CONTRIBUTING.md](CONTRIBUTING.md). Live task status is in the [tracker](https://claude.ai/artifact/Lov9qQxdkCTrFS5uhqqnQy).

### Person A (Chen Han): teacher data

**Your folders:** `reasoning/teacher/`, `scripts/teacher/`, `data/sft/`

1. Pick the teacher models (about 7B), and write `reasoning/teacher/` code that runs them with vLLM on all 6,973 problems in `data/splits/gsm8k_train.jsonl`. Generate a **standard** and a **long** rationale for each problem.
2. Keep only rationales whose final answer is correct. Check them with `reasoning.answer.is_correct`, not a new parser.
3. Make a **concise** rewrite of each long rationale, and check the answer didn't change.
4. Write the four datasets plus `stats.json` to `data/sft/`, in the format in [reasoning/teacher/\_\_init\_\_.py](reasoning/teacher/__init__.py). Use exactly the same problem IDs in every file. **Due Mon 2 Nov.**
5. Read the six papers in `docs/research-question-card.md` in full and check every citation.
6. **Report:** literature review and the Data subsection.

### Person B (Chua Jun Hong): training

**Your folders:** `reasoning/train/`, `scripts/train/`, `configs/`

1. Write the LoRA SFT script in `reasoning/train/`. Build inputs with `prompts.chat_prompt` and targets with `prompts.format_target`, and compute the loss on answer tokens only.
2. **Don't wait for A.** Debug it on GSM8K's human solutions, which are already in `data/splits/gsm8k_train.jsonl` (the `solution` field).
3. Sweep learning rate, epochs and rank (8, 16, 32) on `gsm8k_val`. Save the winning settings in `configs/` and freeze them.
4. From 2 Nov, train 24 adapters: 4 styles × 2 sizes × 3 seeds. Save them to `ckpt/{style}_{size}_s{seed}/`, each with a `train_log.json`, in the format in [reasoning/train/\_\_init\_\_.py](reasoning/train/__init__.py).
5. **Report:** Methods (SFT, LoRA, training setup) and the hyperparameter table.

### Person C (Luo Yuxuan): evaluation

**Your folders:** `reasoning/eval/`, `scripts/eval/`, `results/`

1. **This week:** set up the TC1 environment (below), confirm your QoS with `MyTCinfo`, and submit `scripts/eval/slurm_baselines.sh` to run the baseline matrix on `gsm8k_val`. This is the first run of the vLLM path.
2. Evaluate each of B's adapters on `gsm8k_val` with `sbatch scripts/eval/slurm_students.sh gsm8k_val`: greedy, and 16 samples for self-consistency, with a 4,096-token cap so long-style students are not cut off by our setting. Copy each finished batch's `*.scored.jsonl` into `results/scored/` for Person A's H4 analysis.
3. From 9 Nov, do the final runs on `gsm8k_test`, `svamp` and `gsm_symbolic` with the frozen settings.
4. Make the accuracy-vs-tokens Pareto plot, maj@k curves, diversity and truncation plots, and significance tests (`reasoning/eval/compare.py`). Commit them to `results/` and **freeze them on Thu 12 Nov**.
5. **Report:** Experiments and Results.

### Everyone

- **9–11 Nov:** error taxonomy. Each person labels about 100 wrong answers. All three label the same 30, to check you agree.
- **13–15 Nov:** Introduction, Discussion and the AI Use Statement. Then everyone reads the whole report and checks they can explain every part.
- **Shared code** (`reasoning/data.py`, `prompts.py`, `answer.py`, `stats.py`, `data/splits/`, `tests/`): only change it through a PR that both other people approve. Every result depends on it.
- **Sun 15 Nov:** the leader submits the PDF report and the code ZIP on NTULearn.

## What each folder means

| Folder | What it is | Owner | In git? |
|---|---|---|---|
| `reasoning/` | The project's Python package. Run its modules from the repo root with `python -m reasoning.<module>`. | — | yes |
| `reasoning/data.py`, `prompts.py`, `answer.py`, `stats.py` | Shared core: the data splits, the one prompt format, the one answer parser, statistics helpers | Shared | yes |
| `reasoning/teacher/` | Teacher generation, rejection sampling and concise rewriting. Its `__init__.py` holds the output format. | A | yes |
| `reasoning/train/` | LoRA fine-tuning code. Its `__init__.py` holds the adapter format. | B | yes |
| `reasoning/eval/` | `generate.py` (run a model), `evaluate.py` (score a run), `compare.py` (significance tests) | C | yes |
| `scripts/teacher/`, `scripts/train/`, `scripts/eval/` | Shell and SLURM job scripts for the cluster, one folder per owner | A / B / C | yes |
| `configs/` | The frozen training hyperparameters | B | yes |
| `data/splits/` | Fixed train / val / test / out-of-distribution splits. Everyone uses these exact files. | Shared | yes |
| `data/sft/` | The four training datasets (`long`, `standard`, `concise`, `answer_only`) and `stats.json` | A | yes; `raw/` is not |
| `results/tables/`, `results/figures/` | The small final outputs the report uses: summary JSON, CSV tables, plots | C | yes |
| `notebooks/` | Exploration notebooks, named `<letter>_<topic>.ipynb` so nobody edits the same one | Everyone | yes |
| `docs/assignment/` | The project handout and FAQ | Everyone | yes |
| `docs/research-question-card.md` | Research question, hypotheses H1–H3, success criteria, paper list | Everyone | yes |
| `docs/handbook.docx` | Snapshot of the project handbook | Everyone | yes |
| `docs/report/` | Report figures, final PDF, `references.bib` | Everyone | yes |
| `tests/` | Unit tests. GitHub runs them on every push. | Shared | yes |
| `.github/workflows/` | The automatic test run | Shared | yes |
| `runs/` | Raw model outputs from `reasoning.eval.generate` (large) | C | **no**, stays on the cluster |
| `ckpt/` | Trained LoRA adapters (large) | B | **no**, stays on the cluster |
| `slurm-*.out`, `slurm-*.err` | SLURM job logs, written in the repo root | Everyone | **no** |

## Handoffs

The handoffs between stages are written down as file formats, so nobody waits on anyone's code:

| From → to | What | Format | Due |
|---|---|---|---|
| A → B | `data/sft/{style}.jsonl` | [reasoning/teacher/\_\_init\_\_.py](reasoning/teacher/__init__.py) | Mon 2 Nov |
| B → C | `ckpt/{style}_{size}_s{seed}/` LoRA adapters | [reasoning/train/\_\_init\_\_.py](reasoning/train/__init__.py) | 2–8 Nov, as they finish |
| C → report | `results/tables/`, `results/figures/` | [results/README.md](results/README.md) | Thu 12 Nov |

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
git clone git@github.com:reathanluo/sc4001-group-project.git && cd sc4001-group-project
module load uv
uv venv --seed --python 3.11 .venv       # the job scripts use .venv/ when it exists
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/uv-cache-$USER UV_LINK_MODE=copy   # home storage has no file locks
uv pip install -r requirements.txt
python -m pytest tests                   # must print 26 passed
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

For writing code, running tests and debugging on a few questions. vLLM needs Linux + CUDA, so it is left out here; use the cluster setup above for real runs. [uv](https://docs.astral.sh/uv/) downloads Python 3.11 for you, and `.venv/` is gitignored.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install transformers datasets pytest torch
python -m pytest tests          # should print 26 passed
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
| `gsm8k_test` | 1,319 | **final runs only, from 9 Nov** |
| `svamp` | 1,000 | out-of-distribution eval |
| `gsm_symbolic` | 5,000 | perturbed GSM8K-test templates; robustness and contamination check |
| `exemplars` | 8 | fixed few-shot exemplars (from `gsm8k_train`) |

## Running evaluation

```bash
sbatch scripts/eval/slurm_baselines.sh      # on the TC1 head node: untrained baselines, 1,024-token cap
sbatch scripts/eval/slurm_students.sh gsm8k_val   # trained LoRA students in ckpt/, 4,096-token cap
bash scripts/eval/run_baselines.sh gsm8k_val  # or interactively on a GPU node

python -m reasoning.eval.generate --model Qwen/Qwen2.5-1.5B-Instruct --prompt chat \
    --split gsm8k_val --n 16 --temperature 0.7 --out runs/inst15_sc16.jsonl
python -m reasoning.eval.evaluate runs/inst15_sc16.jsonl
python -m reasoning.eval.compare runs/A.scored.jsonl runs/B.scored.jsonl --k 1
```

To debug locally without a GPU (slow, 5 questions):

```bash
python -m reasoning.eval.generate --backend hf --model Qwen/Qwen2.5-0.5B --prompt fewshot \
    --split gsm8k_val --limit 5 --max-new-tokens 256 --out runs/debug.jsonl
```

## Rules everyone follows

- **Never tune on `gsm8k_test`.** Prompts, hyperparameters, checkpoints and decoding settings are chosen on `gsm8k_val`. `generate.py` warns when the test split is used.
- **One prompt, one target format.** SFT inputs come from `prompts.chat_prompt(tokenizer, question)` and targets from `prompts.format_target(rationale, gold)`. Only the rationale differs between conditions.
- **One parser.** Every answer is scored by `reasoning/answer.py`.
- **Keep raw generations** in `runs/`. Scoring is a separate step, so a parser fix never needs a GPU rerun.
- **Report cost three ways:**
  - generated tokens per question (primary);
  - approximate FLOPs, 2·N·(prompt + generated);
  - wall-clock time.
  
  The 8-shot prompt's prefix is the same for every question, so report its cost separately. On TC1 prefix caching is off (see above), so wall-clock time includes recomputing it.
- **Self-consistency:** 16 samples at T = 0.7. maj@k is computed from subsets of those 16.

## Status (27 Oct 2026)

- The evaluation harness is written and its 26 unit tests pass. Its vLLM path, LoRA serving, `evaluate` and `compare` have run on a TC1 V100 (smoke test, 27 Oct).
- Teacher data (A) is done: `data/sft/` holds the four datasets on 6,547 shared problems, with `stats.json`. How they were made is in `scripts/teacher/README.md`.
- The training package (B) holds its interface contract only.

Dependencies: see `requirements.txt` (vLLM, transformers, datasets, peft, accelerate, pytest).

## Links

- Task tracker: https://claude.ai/artifact/Lov9qQxdkCTrFS5uhqqnQy
- Project handbook: https://claude.ai/artifact/Jsqphj1rrVeXZ6AXEdq3Lo (a Google Docs copy is shared with the team; `docs/handbook.docx` is a snapshot)
