"""Compare old and revised HF validation on ONE saved adapter, without training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..answer import extract_pred
from .sft import read_rows, validate_rows


def compare(reference, candidate):
    if [row["id"] for row in reference] != [row["id"] for row in candidate]:
        raise ValueError("Validation IDs/order do not match")
    return {
        "text_disagreements": sum(a["text"] != b["text"] for a, b in zip(reference, candidate)),
        "answer_disagreements": sum(extract_pred(a["text"])[0] != extract_pred(b["text"])[0]
                                    for a, b in zip(reference, candidate)),
        "correctness_disagreements": sum(a["correct"] != b["correct"] for a, b in zip(reference, candidate)),
        "correctness_disagreement_ids": [a["id"] for a, b in zip(reference, candidate)
                                         if a["correct"] != b["correct"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_new_tokens <= 0:
        parser.error("Limits must be positive")
    rows = read_rows("data/splits/gsm8k_val.jsonl")[:args.limit]
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("Choose a new empty output directory")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        parser.error("Submit this comparison as a GPU job")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                               attn_implementation="eager")
    base.config.use_cache = False  # Same training-time config as the original callback.
    model = PeftModel.from_pretrained(base, args.adapter).to("cuda")
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    summary = {"config": vars(args), "n_val": len(rows), "status": "running", "modes": {}}
    reference = None
    for name, batch_size, use_cache in (
        ("single_no_cache", 1, False),
        ("single_cached", 1, True),
        ("batch8_cached", 8, True),
    ):
        set_seed(0)
        started = time.monotonic()
        def progress(done, total, seconds):
            print(f"{name}: {done}/{total} ({seconds:.1f}s)", flush=True)
        predictions = validate_rows(model, tokenizer, rows, batch_size, args.max_new_tokens,
                                    progress, use_cache=use_cache)
        # Retain the same raw-output shape as C's evaluator, with extra diagnostic fields.
        with (out / f"{name}.jsonl").open("w") as stream:
            for row in predictions:
                stream.write(json.dumps({**row, "samples": [{"text": row["text"],
                    "gen_tokens": row["gen_tokens"], "finish_reason": row["finish_reason"]}]}) + "\n")
        metrics = {"accuracy": sum(row["correct"] for row in predictions) / len(predictions),
                   "seconds": round(time.monotonic() - started, 2)}
        if reference is None:
            reference = predictions
            # Compare the original callback's recorded predictions when available.
            historical = Path(args.adapter) / "validation_epoch_1.jsonl"
            if historical.exists():
                metrics["vs_saved_epoch1"] = compare(read_rows(historical)[:len(rows)], predictions)
        else:
            metrics["vs_single_no_cache"] = compare(reference, predictions)
        summary["modes"][name] = metrics
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{name}: {json.dumps(metrics)}", flush=True)
    summary["status"] = "complete"
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
