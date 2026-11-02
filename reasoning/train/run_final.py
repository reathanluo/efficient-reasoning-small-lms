"""One frozen-protocol run: FP32 training exits before vLLM validation starts.

Uses no GPU imports in this orchestrator. All GPU work runs in child processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .sft import read_rows

MODELS = {"05B": "Qwen/Qwen2.5-0.5B", "15B": "Qwen/Qwen2.5-1.5B"}
STYLES = ("long", "standard", "concise", "answer_only")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verify_inputs(protocol):
    if protocol.get("status") != "frozen" or protocol["epochs"] != 1:
        raise ValueError("Expected a frozen one-epoch protocol")
    if protocol["lora_alpha"] != 2 * protocol["rank"] or protocol["lora_dropout"] != 0.05:
        raise ValueError("Protocol LoRA settings do not match this training implementation")
    for style in STYLES:
        if sha(f"data/sft/{style}.jsonl") != protocol["data_sha256"][style]:
            raise ValueError(f"Frozen dataset changed: {style}")
    if sha("data/splits/gsm8k_val.jsonl") != protocol["validation_sha256"]:
        raise ValueError("Frozen validation split changed")
    if sha("reasoning/prompts.py") != protocol["prompt_source_sha256"]:
        raise ValueError("Shared prompt source changed; review protocol before training")


def verify_adapter(adapter, protocol, style, size, seed):
    adapter = Path(adapter)
    for name in ("adapter_model.safetensors", "adapter_config.json", "train_log.json"):
        if not (adapter / name).is_file():
            raise ValueError(f"Missing saved artifact: {adapter / name}")
    log = json.loads((adapter / "train_log.json").read_text())
    if log["status"] not in ("complete", "trained_pending_validation"):
        raise ValueError(f"Adapter is not ready for scoring: {log['status']}")
    expected = {"model": MODELS[size], "train_file": f"data/sft/{style}.jsonl",
                "seed": seed, "limit": None, "val_limit": None,
                "learning_rate": protocol["learning_rate"], "rank": protocol["rank"],
                "epochs": protocol["epochs"], "precision": protocol["precision"],
                "batch_size": protocol["batch_size"], "grad_accum": protocol["grad_accum"],
                "max_length": protocol["max_length"], "max_new_tokens": protocol["validation_cap"],
                "val_batch_size": protocol["val_batch_size"]}
    for key, value in expected.items():
        if log["config"].get(key) != value:
            raise ValueError(f"Saved {key} differs from frozen protocol")
    if log.get("n_train") != protocol["n_train"] or log.get("n_val") != protocol["n_val"]:
        raise ValueError("Saved train/validation counts do not match")
    if log.get("data_sha256") != {"train": protocol["data_sha256"][style],
                                  "validation": protocol["validation_sha256"]}:
        raise ValueError("Saved input hashes do not match frozen data")
    lora = json.loads((adapter / "adapter_config.json").read_text())
    for key, expected_value in (("r", protocol["rank"]), ("lora_alpha", protocol["lora_alpha"]),
                                ("lora_dropout", protocol["lora_dropout"]), ("bias", "none")):
        if lora.get(key) != expected_value:
            raise ValueError(f"Saved LoRA {key} differs")
    if set(lora.get("target_modules", [])) != set(protocol["target_modules"]):
        raise ValueError("Saved target modules differ")
    return log


def attach_validation(log, rows, scored, summary, protocol, source):
    expected = read_rows("data/splits/gsm8k_val.jsonl")
    if [(r["id"], r["gold"]) for r in rows] != [(r["id"], r["gold"]) for r in expected]:
        raise ValueError("Validation IDs/gold/order differ from frozen split")
    if len(scored) != len(expected) or [r["id"] for r in scored] != [r["id"] for r in expected]:
        raise ValueError("Scored validation IDs do not match")
    if any(len(r["samples"]) != 1 for r in rows):
        raise ValueError("Expected one greedy sample per validation question")
    if summary["n_questions"] != protocol["n_val"] or summary["n_samples"] != 1:
        raise ValueError("Evaluation sample counts differ")
    if abs(sum(bool(r["correct"][0]) for r in scored) / len(scored) - summary["sample_acc"]) > 1e-9:
        raise ValueError("Summary accuracy differs from scored predictions")
    epoch = next((item for item in log["epochs"] if item["epoch"] == 1), None)
    if epoch is None:
        raise ValueError("Training epoch log is missing")
    # Retain the original HF diagnostic instead of relabelling it as a vLLM score.
    if epoch.get("validation_backend", "hf") != "vllm" and epoch.get("gsm8k_val_accuracy") is not None:
        epoch["hf_gsm8k_val_accuracy"] = epoch["gsm8k_val_accuracy"]
        epoch["hf_validation_seconds"] = epoch.get("validation_seconds")
    epoch.update(gsm8k_val_accuracy=summary["sample_acc"], validation_backend="vllm",
                 validation_status="complete", validation_seconds=summary.get("wall_seconds"))
    log.update(status="complete", stage="complete", final_protocol_id=protocol["protocol_id"],
               final_validation={"raw": str(source), "summary": str(source.with_suffix('.summary.json')),
                                 "cap": protocol["validation_cap"], "dtype": protocol["validation_dtype"]})
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("style", choices=STYLES)
    parser.add_argument("size", choices=MODELS)
    parser.add_argument("seed", type=int)
    parser.add_argument("--config", default="configs/sft-final.json")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path(args.config).read_text())
    verify_inputs(protocol)
    if args.seed not in protocol["seeds"]:
        parser.error("Seed is not in frozen protocol")
    run_id = f"{args.style}_{args.size}_s{args.seed}"
    reused = run_id in protocol["pilot_reuse"]
    adapter = Path(protocol["pilot_reuse"].get(run_id, f"ckpt/{run_id}"))
    attempt = os.environ.get("SLURM_JOB_ID", str(time.time_ns()))
    raw = Path("runs/final_validation") / run_id / f"{attempt}.jsonl"
    train = [sys.executable, "-m", "reasoning.train.sft", "--model", MODELS[args.size],
             "--train-file", f"data/sft/{args.style}.jsonl", "--out", str(adapter),
             "--precision", protocol["precision"], "--defer-validation"]
    for flag, key in (("learning-rate", "learning_rate"), ("rank", "rank"), ("epochs", "epochs"),
                      ("batch-size", "batch_size"), ("grad-accum", "grad_accum"),
                      ("max-length", "max_length"), ("val-batch-size", "val_batch_size"),
                      ("max-new-tokens", "validation_cap"), ("save-steps", "save_steps")):
        train.extend([f"--{flag}", str(protocol[key])])
    train.extend(["--seed", str(args.seed)])
    if args.resume_from_checkpoint:
        if reused:
            parser.error("Reuse pilots as completed adapters; recover them with their original script")
        train.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])
    generate = [sys.executable, "-m", "reasoning.eval.generate", "--model", MODELS[args.size],
                "--lora", str(adapter), "--prompt", "chat", "--split", "gsm8k_val", "--n", "1",
                "--temperature", "0", "--seed", str(protocol["validation_seed"]),
                "--dtype", protocol["validation_dtype"], "--max-model-len", str(protocol["validation_max_model_len"]),
                "--max-new-tokens", str(protocol["validation_cap"]), "--out", str(raw)]
    score = [sys.executable, "-m", "reasoning.eval.evaluate", str(raw)]
    if args.dry_run:
        print(f"{run_id}: {'verify and reuse pilot' if reused else 'train if not already saved'}: {adapter}")
        for command in ([] if reused else [train]) + [generate, score]:
            print(shlex.join(command))
        return
    if reused and not adapter.exists():
        parser.error(f"Pilot missing here: {adapter}; arrange access/copy, do not duplicate it blindly")
    ready = adapter.is_dir() and (adapter / "train_log.json").is_file()
    if ready and not args.resume_from_checkpoint:
        verify_adapter(adapter, protocol, args.style, args.size, args.seed)
        print(f"Reusing verified adapter: {adapter}", flush=True)
    else:
        subprocess.run(train, check=True)
    log = verify_adapter(adapter, protocol, args.style, args.size, args.seed)
    # The training child has exited; GPU weights/optimizer memory are released.
    subprocess.run(generate, check=True)
    subprocess.run(score, check=True)
    rows, scored = read_rows(raw), read_rows(raw.with_suffix(".scored.jsonl"))
    summary = json.loads(raw.with_suffix(".summary.json").read_text())
    updated = attach_validation(log, rows, scored, summary, protocol, raw)
    # Write C's expected per-epoch predictions beside the adapter as well as raw runs.
    with (adapter / "validation_epoch_1_vllm.jsonl").open("w") as stream:
        for row, marked in zip(rows, scored):
            stream.write(json.dumps({"id": row["id"], "gold": row["gold"],
                "text": row["samples"][0]["text"], "correct": bool(marked["correct"][0]),
                "gen_tokens": row["samples"][0]["gen_tokens"],
                "finish_reason": row["samples"][0]["finish_reason"]}) + "\n")
    atomic_json(adapter / "train_log.json", updated)
    record = {"run_id": run_id, "status": "evaluated", "protocol_id": protocol["protocol_id"],
              "protocol_sha256": sha(args.config), "adapter": str(adapter),
              "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
              "base_model": MODELS[args.size], "style": args.style, "size": args.size,
              "seed": args.seed, "data_sha256": protocol["data_sha256"][args.style],
              "validation": str(raw), "accuracy": summary["sample_acc"],
              "vllm_wall_seconds": summary.get("wall_seconds"),
              "adapter_job_wall_seconds": log.get("wall_seconds"),
              "adapter_job_includes_legacy_hf_validation": reused,
              "validation_cap": protocol["validation_cap"], "job_id": os.environ.get("SLURM_JOB_ID"),
              "reused_pilot": reused, "backup_verified": False, "handed_off": False}
    atomic_json(Path("scripts/train/records/final_runs") / f"{run_id}.json", record)
    print(f"FINAL RUN COMPLETE: {run_id}, accuracy={summary['sample_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
