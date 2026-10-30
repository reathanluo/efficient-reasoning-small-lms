"""Completion-only LoRA SFT. Run from the repo root with --help for options."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from ..answer import extract_pred, is_correct
from ..prompts import chat_prompt, format_target


def check_training_metrics(metrics):
    for name in ("loss", "grad_norm"):
        if name in metrics and not math.isfinite(float(metrics[name])):
            raise FloatingPointError(
                f"Non-finite {name}: {metrics[name]}. Stop this run; "
                "check precision and model/data before continuing."
            )


def load_resume_log(out, checkpoint, config, data_hashes):
    """Reject incomplete checkpoints, changed inputs, and incompatible settings."""
    out, checkpoint = Path(out), Path(checkpoint)
    if checkpoint.resolve().parent != out.resolve():
        raise ValueError("Resume checkpoint must be directly inside --out")
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if config["precision"] == "float16" and not (checkpoint / "scaler.pt").is_file():
        missing.append("scaler.pt")
    if missing:
        raise ValueError(f"Incomplete checkpoint: missing {missing}")
    log = json.loads((out / "train_log.json").read_text())
    immutable = ("model", "train_file", "limit", "val_limit", "val_batch_size", "epochs",
                 "rank", "learning_rate", "precision", "batch_size", "grad_accum",
                 "max_length", "max_new_tokens", "seed")
    for key in immutable:
        if log["config"].get(key) != config.get(key):
            raise ValueError(f"Cannot resume with changed {key}")
    if log.get("data_sha256") != data_hashes:
        raise ValueError("Input hashes differ or are missing; do not resume changed data")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    log["epochs"] = [row for row in log["epochs"] if row["global_step"] <= state["global_step"]]
    log.setdefault("resume_events", []).append({"checkpoint": str(checkpoint),
                                               "global_step": state["global_step"]})
    log.pop("error", None)
    log.pop("validation_progress", None)
    log["status"] = "running"
    log["stage"] = "training"
    return log


def read_rows(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def check_data(train, val):
    """Only committed GSM8K training IDs may enter SFT."""
    allowed = {row["id"] for row in read_rows("data/splits/gsm8k_train.jsonl")}
    if any(row["id"] not in allowed for row in train):
        raise ValueError("Training data contains IDs outside gsm8k_train")
    if {row["id"] for row in train} & {row["id"] for row in val}:
        raise ValueError("Training and validation IDs overlap")
    for row in train:
        for key in ("question", "gold"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Invalid {key} for {row['id']}")
        rationale = row.get("rationale", row.get("solution"))
        if not isinstance(rationale, str) or "####" in rationale:
            raise ValueError(f"Expected rationale without final answer: {row['id']}")


def encode_row(row, tokenizer, max_length):
    prompt = chat_prompt(tokenizer, row["question"])
    rationale = row.get("rationale", row.get("solution"))
    target = format_target(rationale, row["gold"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    target_ids = target_ids + [tokenizer.eos_token_id]
    ids = prompt_ids + target_ids
    # Never silently remove the final answer or select different examples by style.
    if len(ids) > max_length:
        raise ValueError(
            f"{row['id']} needs {len(ids)} tokens, above max_length={max_length}. "
            "Increase the limit consistently across styles; no truncation was applied."
        )
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + target_ids}


def pad_batch(features, pad_id):
    """Right padding; mask by position even when PAD and EOS share an ID."""
    length = max(len(item["input_ids"]) for item in features)
    batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for item in features:
        padding = length - len(item["input_ids"])
        for key, value in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)):
            batch[key].append(item[key] + [value] * padding)
    return batch


def validate_rows(model, tokenizer, rows, batch_size, max_new_tokens, progress=None, use_cache=True):
    """Greedy, left-padded generation; restore training state even on failure."""
    import torch

    was_training = model.training
    padding_side = tokenizer.padding_side
    predictions = []
    started = time.monotonic()
    try:
        model.eval()
        tokenizer.padding_side = "left"
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start:start + batch_size]
                prompts = [chat_prompt(tokenizer, row["question"]) for row in batch_rows]
                encoded = tokenizer(prompts, padding=True, add_special_tokens=False,
                                    return_tensors="pt").to(model.device)
                generated = model.generate(
                    **encoded, do_sample=False, use_cache=use_cache,
                    max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                width = encoded["input_ids"].shape[1]
                for row, sequence in zip(batch_rows, generated):
                    tokens = sequence[width:].tolist()
                    # Batched generate pads shorter completions after their EOS.
                    if tokenizer.eos_token_id in tokens:
                        tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    pred, method = extract_pred(text)
                    predictions.append({"id": row["id"], "gold": row["gold"], "text": text,
                                        "correct": is_correct(pred, row["gold"]), "method": method,
                                        "gen_tokens": len(tokens),
                                        "finish_reason": "stop" if tokens and tokens[-1] == tokenizer.eos_token_id else "length"})
                if progress:
                    progress(len(predictions), len(rows), time.monotonic() - started)
    finally:
        tokenizer.padding_side = padding_side
        model.train(was_training)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train-file", default="data/splits/gsm8k_train.jsonl")
    parser.add_argument("--out", default="ckpt/human_05B_s0_smoke")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--val-batch-size", type=int, default=1,
                        help="Keep 1 for tuning: batch 8 changed validation answers on TC1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, choices=[8, 16, 32], default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16",
                        help="Use float32 to diagnose non-finite FP16 training on V100")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-steps", type=int, help="Also checkpoint every N optimizer steps")
    parser.add_argument("--resume-from-checkpoint", help="Complete Trainer checkpoint under --out")
    parser.add_argument("--check-data", action="store_true", help="Validate JSONL only; no model downloads")
    args = parser.parse_args()
    for key in ("limit", "val_limit", "val_batch_size", "epochs", "batch_size", "grad_accum", "max_length", "max_new_tokens", "save_steps"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    train_rows = read_rows(args.train_file)
    val_rows = read_rows("data/splits/gsm8k_val.jsonl")
    check_data(train_rows, val_rows)
    train_rows = train_rows[:args.limit]
    val_rows = val_rows[:args.val_limit]
    print(f"Validated {len(train_rows)} training and {len(val_rows)} validation examples", flush=True)
    if args.check_data:
        return
    out = Path(args.out)
    if not args.resume_from_checkpoint and out.exists() and any(out.iterdir()):
        parser.error(f"Output directory is not empty: {out}; choose a new --out")
    hashes = {"train": hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
              "validation": hashlib.sha256(Path("data/splits/gsm8k_val.jsonl").read_bytes()).hexdigest()}
    resume_log = None
    if args.resume_from_checkpoint:
        try:
            resume_log = load_resume_log(out, args.resume_from_checkpoint, vars(args), hashes)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainerCallback, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Run training in a GPU job on TC1. Use --check-data locally.")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        raise ValueError("Base tokenizer has no chat template. Agree a shared template with C before training.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = [encode_row(row, tokenizer, args.max_length) for row in train_rows]
    model_dtype = torch.float16 if args.precision == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=model_dtype,
                                                attn_implementation="eager")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.rank, lora_alpha=2 * args.rank,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log = resume_log if resume_log is not None else {
        "config": vars(args), "n_train": len(dataset), "n_val": len(val_rows),
        "data_sha256": hashes, "epochs": [], "status": "running", "stage": "training",
        "validation_predictions": "validation_epoch_*.jsonl"}
    previous_wall = log.get("wall_seconds", 0)

    def write_log():
        log["wall_seconds"] = round(previous_wall + time.monotonic() - started, 2)
        temporary = out / "train_log.json.tmp"
        temporary.write_text(json.dumps(log, indent=2) + "\n")
        temporary.replace(out / "train_log.json")

    class Validate(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            check_training_metrics(logs or {})

        def on_save(self, args, state, control, **kwargs):
            log["last_checkpoint"] = str(out / f"checkpoint-{state.global_step}")
            write_log()

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            epoch = round(state.epoch)
            def progress(done, total, seconds):
                log["stage"] = "validation"
                log["validation_progress"] = {"epoch": epoch, "done": done, "total": total,
                                              "seconds": round(seconds, 2)}
                write_log()
                print(f"Epoch {epoch}: validation {done}/{total} ({seconds:.1f}s)", flush=True)

            print(f"Epoch {epoch}: starting validation on {len(val_rows)} questions", flush=True)
            validation_started = time.monotonic()
            predictions = validate_rows(model, tokenizer, val_rows, main_args.val_batch_size,
                                        main_args.max_new_tokens, progress)
            (out / f"validation_epoch_{epoch}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions))
            accuracy = sum(row["correct"] for row in predictions) / len(predictions)
            # Trainer restores log_history on resume, including earlier steps in this epoch.
            losses = [item["loss"] for item in state.log_history
                      if "loss" in item and epoch - 1 < item.get("epoch", -1) <= epoch + 1e-6]
            log["epochs"] = [item for item in log["epochs"] if item["epoch"] != epoch]
            log["epochs"].append({"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)),
                                  "gsm8k_val_accuracy": accuracy, "global_step": state.global_step,
                                  "validation_seconds": round(time.monotonic() - validation_started, 2)})
            log["stage"] = "training"
            print(f"Epoch {epoch}: validation accuracy={accuracy:.3f}", flush=True)
            control.should_save = True
            write_log()

    def collate(features):
        return {key: torch.tensor(values, dtype=torch.long)
                for key, values in pad_batch(features, tokenizer.pad_token_id).items()}

    main_args = args
    trainer = Trainer(
        model=model, train_dataset=dataset, data_collator=collate,
        args=TrainingArguments(
            output_dir=str(out), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
            fp16=args.precision == "float16", gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_strategy="steps", logging_steps=1,
            save_strategy="steps" if args.save_steps else "epoch",
            save_steps=args.save_steps or 500, save_total_limit=2,
            logging_nan_inf_filter=False,
            report_to="none", seed=args.seed, data_seed=args.seed, label_names=["labels"],
        ), callbacks=[Validate()],
    )
    write_log()
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except Exception as error:
        log["status"] = "failed"
        log["stage"] = "failed"
        log["error"] = f"{type(error).__name__}: {error}"
        log["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        write_log()
        raise
    log["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    log["peak_torch_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    log["status"] = "complete"
    log["stage"] = "complete"
    log["history"] = trainer.state.log_history
    write_log()
    print(f"Adapter saved to {out}")
    print(f"Peak PyTorch GPU allocation: {log['peak_torch_allocated_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
