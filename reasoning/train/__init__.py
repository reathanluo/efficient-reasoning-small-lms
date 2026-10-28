"""LoRA supervised fine-tuning (owner: Person B).

The starter implementation is `sft.py`; see scripts/train/README.md for the
local checks and TC1 smoke job. The contracts below are what A and C rely on.

Input
-----
data/sft/{style}.jsonl from Person A (format in reasoning/teacher/__init__.py).
Until those exist, develop on GSM8K's human solutions: data/splits/gsm8k_train.jsonl,
using its `solution` field as the rationale.

Formatting (must match evaluation exactly)
------------------------------------------
    prompt = reasoning.prompts.chat_prompt(tokenizer, question)
    target = reasoning.prompts.format_target(rationale, gold)
Compute the loss on the target tokens only (mask the prompt tokens).

Output: ckpt/{style}_{size}_s{seed}/   e.g. ckpt/concise_05B_s0/
------------------------------------------------------------------
* A PEFT LoRA adapter directory (adapter_config.json + adapter weights) that
  reasoning.eval.generate loads with --lora. Don't merge it into the base model.
* train_log.json beside it: the config used, plus train loss and gsm8k_val
  accuracy per epoch (Person C uses these for the overfitting curves).
* ckpt/ is gitignored. Copy finished adapters to the shared cluster folder.

Rules
-----
* Choose hyperparameters on gsm8k_val only, record them in configs/, then freeze
  them for every style, size and seed.
* Put cluster job scripts in scripts/train/.
"""
