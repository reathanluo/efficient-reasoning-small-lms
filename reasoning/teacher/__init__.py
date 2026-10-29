"""Teacher data: rationale generation and the four SFT datasets (owner: Person A).

Modules: generate.py (teacher sampling), filter.py (rejection sampling),
compress.py (long -> concise), build.py (the four files and stats.json). The
commands that produced data/sft/ are in scripts/teacher/README.md. The OUTPUT
CONTRACT below is what Person B's training code reads, so keep to it or agree a
change first.

Input
-----
data/splits/gsm8k_train.jsonl   (6,973 problems: id, question, gold, solution)
Never read gsm8k_val, gsm8k_test, svamp or gsm_symbolic here.

Output: data/sft/{style}.jsonl, style in {long, standard, concise, answer_only}
----------------------------------------------------------------------------
One JSON object per line:
    {"id": "gsm8k-train-1166", "question": "...", "gold": "34",
     "rationale": "Day 1: 70 * 0.30 = 21 ...", "style": "concise"}

Rules
-----
* All four files contain EXACTLY the same set of ids, in the same order.
* Keep only rationales whose final answer equals `gold`; check it with
  reasoning.answer.extract_pred / is_correct, never with a new parser.
* `rationale` is the working only. Don't add the "#### answer" line yourself;
  prompts.format_target(rationale, gold) appends it at training time.
* answer_only rows have rationale = "".
* Also write data/sft/stats.json: per style, the count and the mean, median and
  90th-percentile rationale length in Qwen2.5 tokens.

Suggested modules: generate.py (teacher sampling with vLLM), filter.py
(rejection sampling), compress.py (long -> concise rewrite), build.py (writes
the four files and stats.json). Put cluster job scripts in scripts/teacher/.
"""
