"""Generate model outputs for an evaluation split and save them raw.

Generation and scoring are separate steps: this script only samples and records
text, token counts and finish reasons; `reasoning.eval.evaluate` scores the file.
That way the parser can be fixed later without re-running the GPU jobs.

Examples
--------
Greedy, base model, 8-shot (cluster, vLLM):
    python -m reasoning.eval.generate --model Qwen/Qwen2.5-0.5B --prompt fewshot \
        --split gsm8k_val --out runs/base05_fewshot_greedy.jsonl

Self-consistency samples (16 per question, T=0.7):
    python -m reasoning.eval.generate --model Qwen/Qwen2.5-0.5B-Instruct --prompt chat \
        --split gsm8k_val --n 16 --temperature 0.7 --out runs/inst05_chat_sc16.jsonl

SFT student (LoRA adapter, served by vLLM without merging):
    python -m reasoning.eval.generate --model Qwen/Qwen2.5-0.5B --lora ckpt/concise_s0 \
        --prompt chat --split gsm8k_val --out runs/concise_s0_greedy.jsonl

Local debugging without a GPU uses `--backend hf --limit 5`. Only vLLM outputs
are reported in the paper.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

from ..data import load_split
from ..prompts import build_prompt, stop_strings


def count_params(model_name: str) -> int:
    """Parameter count from the config alone (no weights loaded)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return sum(p.numel() for p in model.parameters())


def generate_vllm(args, prompts: List[str]) -> List[List[Dict]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_lora=args.lora is not None,
        max_lora_rank=args.max_lora_rank,
        enable_prefix_caching=True,  # the shared few-shot prefix is computed once
    )
    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_new_tokens,
        stop=stop_strings(args.prompt),
        seed=args.seed,
    )
    lora = LoRARequest("student", 1, args.lora) if args.lora else None
    outputs = llm.generate(prompts, params, lora_request=lora)
    return [
        [
            {
                "text": o.text,
                "gen_tokens": len(o.token_ids),
                "finish_reason": o.finish_reason,  # "stop" or "length" (= truncated)
            }
            for o in out.outputs
        ]
        for out in outputs
    ]


def generate_hf(args, prompts: List[str], tokenizer) -> List[List[Dict]]:
    """Slow single-prompt fallback for local debugging (CPU / Apple MPS)."""
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    if args.lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    torch.manual_seed(args.seed)

    stops = stop_strings(args.prompt)
    results = []
    for i, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p if args.temperature > 0 else None,
                num_return_sequences=args.n,
                stop_strings=stops or None,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        samples = []
        for seq in out[:, enc["input_ids"].shape[1]:]:
            ids = [t for t in seq.tolist() if t != tokenizer.pad_token_id]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            hit_stop = any(s in text for s in stops)
            for s in stops:
                text = text.split(s, 1)[0]
            hit_eos = tokenizer.eos_token_id in ids
            samples.append({
                "text": text,
                "gen_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                "finish_reason": "stop" if (hit_stop or hit_eos or len(ids) < args.max_new_tokens) else "length",
            })
        results.append(samples)
        print(f"[hf] {i + 1}/{len(prompts)}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path (the base model when using --lora)")
    ap.add_argument("--lora", default=None, help="path to a LoRA adapter (SFT students)")
    ap.add_argument("--prompt", choices=["fewshot", "chat"], required=True)
    ap.add_argument("--split", required=True, help="gsm8k_val | gsm8k_test | svamp | gsm_symbolic")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per question (1 = greedy when T=0)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--dtype", default="auto", help="use float16 on GPUs without bfloat16 (e.g. V100)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    args = ap.parse_args()

    if args.split == "gsm8k_test":
        print("WARNING: generating on the GSM8K TEST split. Final runs only.", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_split(args.split, args.splits_dir)[: args.limit]
    exemplars = load_split("exemplars", args.splits_dir) if args.prompt == "fewshot" else None
    prompts = [build_prompt(args.prompt, r["question"], tokenizer, exemplars) for r in rows]
    prompt_tokens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]

    t0 = time.time()
    if args.backend == "vllm":
        samples = generate_vllm(args, prompts)
    else:
        samples = generate_hf(args, prompts, tokenizer)
    wall = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r, p_tok, s in zip(rows, prompt_tokens, samples):
            f.write(json.dumps({"id": r["id"], "gold": r["gold"], "prompt_tokens": p_tok, "samples": s}) + "\n")

    meta = {
        **vars(args),
        "n_params": count_params(args.model),
        "n_questions": len(rows),
        "wall_seconds": round(wall, 2),
        "host": platform.node(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(rows)} questions x {args.n} samples to {out} in {wall:.1f}s")


if __name__ == "__main__":
    main()
