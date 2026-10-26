"""Prompt formats shared by evaluation and SFT.

Two formats:
  * "fewshot": plain-text 8-shot CoT for *base* models (no chat template).
  * "chat"   : the tokenizer's chat template with one fixed instruction. Used for
               Instruct models and for every SFT student. P2's training code
               must build its inputs with `chat_prompt` so that training and
               evaluation see identical prompts.

The instruction is deliberately neutral about length ("solve", not "think step
by step"): the rationale style is the variable we manipulate through training,
so the prompt must not push it in either direction.
"""
from __future__ import annotations

from typing import Dict, List

INSTRUCTION = (
    "Solve the following math problem. "
    "Write your final answer on the last line in the form '#### <number>'."
)

# Stop the base model before it invents the next few-shot question.
FEWSHOT_STOP = ["\n\nQuestion:", "\nQuestion:"]


def format_target(solution: str, gold: str) -> str:
    """Canonical completion: rationale, then '#### answer'. Shared with SFT data."""
    solution = solution.strip()
    return f"{solution}\n#### {gold}" if solution else f"#### {gold}"


def fewshot_prompt(question: str, exemplars: List[Dict]) -> str:
    shots = "".join(
        f"Question: {e['question']}\nAnswer: {format_target(e['solution'], e['gold'])}\n\n"
        for e in exemplars
    )
    return f"{shots}Question: {question}\nAnswer:"


def chat_messages(question: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}]


def chat_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        chat_messages(question), tokenize=False, add_generation_prompt=True
    )


def build_prompt(mode: str, question: str, tokenizer=None, exemplars=None) -> str:
    if mode == "fewshot":
        return fewshot_prompt(question, exemplars)
    if mode == "chat":
        return chat_prompt(tokenizer, question)
    raise ValueError(f"unknown prompt mode {mode!r}")


def stop_strings(mode: str) -> List[str]:
    return FEWSHOT_STOP if mode == "fewshot" else []
