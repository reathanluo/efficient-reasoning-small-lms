# SC4001 Group Project: Efficient Reasoning with Small LMs (Topic G)

**Research question:** for Qwen2.5-0.5B/1.5B on grade-school maths (GSM8K), is a fixed token budget better spent:

- in training, by LoRA-distilling long, standard or concise rationales from a teacher, or
- at inference, by self-consistency (majority vote over several samples)?

And do the two compose? The hypotheses and success criteria are in [docs/research-question-card.md](docs/research-question-card.md).

**Deadline:** Mon 16 Nov 2026, 11:59 PM. We aim to submit on Sun 15 Nov.

## Who owns what

Each person owns one stage of the pipeline: the folders below, and the report sections that describe them. You can read and run anything. To change a folder you don't own, open a pull request and tag its owner. Shared files need approval from both other members. The workflow is in [CONTRIBUTING.md](CONTRIBUTING.md).

| Folder | Owner | What lives there |
|---|---|---|
| `reasoning/teacher/`, `scripts/teacher/`, `data/sft/` | **Person A**: teacher data | Teacher generation, rejection sampling, concise rewrites, the 4 SFT datasets |
| `reasoning/train/`, `scripts/train/`, `configs/` | **Person B**: training | LoRA SFT script, hyperparameter configs, adapter training jobs |
| `reasoning/eval/`, `scripts/eval/`, `results/` | **Person C**: evaluation | Generation, scoring, statistics, baselines, final tables and figures |
| `reasoning/{data,prompts,answer,stats}.py`, `data/splits/`, `tests/` | **Shared**: needs both others' approval | Splits, prompt format, answer parser, statistics. Every condition depends on these, so a change affects everyone's results. |
| `docs/`, `notebooks/` | **Everyone** | Planning docs, assignment files, report assets, exploration notebooks |

The handoffs between stages are written down as contracts, so nobody waits on anyone's code:

- **A → B:** `data/sft/{style}.jsonl`. The format is in [reasoning/teacher/\_\_init\_\_.py](reasoning/teacher/__init__.py). Due **Mon 2 Nov**.
- **B → C:** `ckpt/{style}_{size}_s{seed}/` LoRA adapters. The format is in [reasoning/train/\_\_init\_\_.py](reasoning/train/__init__.py).
- **C → report:** `results/tables/` and `results/figures/`. Frozen **Thu 12 Nov**.

## Layout

```
reasoning/            Python package (run modules with python -m from the repo root)
  data.py prompts.py answer.py stats.py   shared core
  teacher/            Person A
  train/              Person B
  eval/               Person C: generate.py, evaluate.py, compare.py
scripts/{teacher,train,eval}/   SLURM / shell job scripts, by owner
configs/              frozen training hyperparameters (B)
data/splits/          fixed train/val/test/OOD splits (committed, shared)
data/sft/             the 4 SFT datasets (A)
results/              small committed outputs used in the report (C)
notebooks/            exploration, one file per person
docs/                 assignment PDFs, research card, handbook, report assets
tests/                unit tests (run by GitHub Actions on every push)
runs/ ckpt/ logs/     large outputs: gitignored, kept on the cluster
```

## Setup (NTU TC1 cluster)

```bash
git clone <repo-url> && cd sc4001-group-project
conda create -n sc4001-g python=3.11 -y
conda activate sc4001-g
pip install -r requirements.txt
python -m pytest tests          # must pass before any run

# On the login node (compute nodes may have no internet), pre-download the models:
export HF_HOME=$HOME/hf_cache
for m in Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct; do
  huggingface-cli download $m
done
```

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
sbatch scripts/eval/slurm_baselines.sh      # edit its ADAPT lines first
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
  
  The 8-shot prompt's prefix is cached across questions, so its cost is mostly amortised.
- **Self-consistency:** 16 samples at T = 0.7. maj@k is computed from subsets of those 16.

## Status (27 Oct 2026)

- The evaluation harness is written and its 26 unit tests pass. It has run end to end on a laptop (HF backend). **The vLLM path has not yet run on the cluster.**
- The teacher (A) and training (B) packages hold interface contracts only.

Dependencies: see `requirements.txt` (vLLM, transformers, datasets, peft, accelerate, pytest).

## Links

- Task tracker: https://claude.ai/artifact/Lov9qQxdkCTrFS5uhqqnQy
- Project handbook: https://claude.ai/artifact/Jsqphj1rrVeXZ6AXEdq3Lo (a Google Docs copy is shared with the team; `docs/handbook.docx` is a snapshot)
