"""Copy small completed/failed training and evaluation evidence; never copy weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def collect(ckpt_dir, runs_dir, out):
    ckpt_dir, runs_dir, out = map(Path, (ckpt_dir, runs_dir, out))
    if out.resolve().is_relative_to(ckpt_dir.resolve()) or out.resolve().is_relative_to(runs_dir.resolve()):
        raise ValueError("Records output must be outside source checkpoint/run directories")
    out.mkdir(parents=True, exist_ok=True)
    index = []

    def copy(source, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for path in sorted(ckpt_dir.glob("*/train_log.json")):
        log = json.loads(path.read_text())
        if log.get("status") not in ("complete", "failed"):
            continue
        relative = path.relative_to(ckpt_dir)
        copy(path, Path("checkpoints") / relative)
        adapter_config = path.parent / "adapter_config.json"
        if adapter_config.is_file():
            copy(adapter_config, Path("checkpoints") / adapter_config.relative_to(ckpt_dir))
        config = log.get("config", {})
        epochs = log.get("epochs", [])
        last = epochs[-1] if epochs else {}
        index.append({"run": path.parent.name, "status": log["status"],
                      "model": config.get("model", ""), "train_file": config.get("train_file", ""),
                      "precision": config.get("precision", "not_recorded"),
                      "seed": config.get("seed", ""), "rank": config.get("rank", ""),
                      "learning_rate": config.get("learning_rate", ""),
                      "requested_epochs": config.get("epochs", ""),
                      "last_recorded_epoch": last.get("epoch", ""),
                      "last_hf_val_accuracy": last.get("gsm8k_val_accuracy", ""),
                      "wall_seconds": log.get("wall_seconds", ""),
                      "error": log.get("error", "")})
    # Explicit allowlist: no raw generations, predictions, optimizer states or weights.
    patterns = ("*.summary.json", "*.meta.json", "teacher_audit_*.json", "summary.json", "*.selection.json")
    sources = {path for pattern in patterns for path in runs_dir.rglob(pattern) if path.is_file()}
    for path in sorted(sources):
        json.loads(path.read_text())  # Validate rather than silently archiving malformed JSON.
        copy(path, Path("evaluation") / path.relative_to(runs_dir))
    fields = ["run", "status", "model", "train_file", "precision", "seed", "rank", "learning_rate",
              "requested_epochs", "last_recorded_epoch", "last_hf_val_accuracy", "wall_seconds", "error"]
    with (out / "training-index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index)
    checksums = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out)}\n")
    (out / "SHA256SUMS").write_text("".join(checksums))
    return {"training_logs": len(index), "evaluation_records": len(sources), "output": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="scripts/train/records")
    args = parser.parse_args()
    if not Path(args.ckpt_dir).is_dir() or not Path(args.runs_dir).is_dir():
        parser.error("Expected checkpoint and run directories; run this in the TC1 checkout")
    print(json.dumps(collect(args.ckpt_dir, args.runs_dir, args.out), indent=2))


if __name__ == "__main__":
    main()
