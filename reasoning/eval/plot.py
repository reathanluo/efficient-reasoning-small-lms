"""Plot accuracy against generated tokens, one panel per model size.

    python -m reasoning.eval.plot results/tables \
        --out results/figures/accuracy_vs_tokens_gsm8k_val \
        --csv results/tables/students_mean_gsm8k_val.csv

Reads the students' *.summary.json files and averages over training seeds. Each
style is a voting curve (k = 1, 2, 4, 8, 16 at T = 0.7) plus one greedy answer.
Also draws the Instruct baselines when their summaries are in the same folder.
Writes <out>.png and <out>.pdf, and a CSV of every plotted point.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RUN_RE = re.compile(r"(answer_only|concise|standard|long)_(05B|15B)_s\d(?:_pilot_\d+)?_(greedy|sc16)$")
SIZES = {"05B": "Qwen2.5-0.5B", "15B": "Qwen2.5-1.5B"}
STYLES = {  # fixed colour per style, in legend order
    "standard": ("Standard", "#2a78d6"),
    "concise": ("Concise", "#eb6834"),
    "long": ("Long", "#1baf7a"),
    "answer_only": ("Answer-only", "#eda100"),
}
INSTRUCT = ("Instruct baseline", "#8a8984")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BUDGET = 512  # the token budget H1 is about


def load_points(folder: Path) -> list[dict]:
    """One row per (size, style, run, k): accuracy and tokens averaged over seeds."""
    groups = defaultdict(list)
    for path in sorted(folder.glob("*.summary.json")):
        summary = json.loads(path.read_text())
        stem = path.name.removesuffix(".summary.json")
        if m := RUN_RE.match(stem):
            style, size, run = m.groups()
        elif m := re.fullmatch(r"inst(05B|15B)_chat_(greedy|sc16)", stem):
            style, size, run = "instruct", m.group(1), m.group(2)
        else:
            continue
        for point in summary["curve"]:
            groups[size, style, run, point["k"]].append(point | {"trunc": summary["truncation_rate"]})

    rows = []
    for (size, style, run, k), points in sorted(groups.items()):
        accs = [p["acc"] for p in points]
        rows.append({
            "size": size, "style": style, "run": run, "k": k, "n_seeds": len(points),
            "acc": round(statistics.mean(accs), 4),
            "acc_sd": round(statistics.stdev(accs), 4) if len(accs) > 1 else "",
            "gen_tokens": round(statistics.mean(p["gen_tokens"] for p in points), 1),
            "truncation_rate": round(statistics.mean(p["trunc"] for p in points), 4),
        })
    return rows


def draw(ax, rows: list[dict], size: str) -> None:
    def series(style: str, run: str) -> list[dict]:
        return [r for r in rows if r["size"] == size and r["style"] == style and r["run"] == run]

    ax.axvline(BUDGET, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    for style, (label, color) in [("instruct", INSTRUCT), *STYLES.items()]:  # baseline underneath
        votes, greedy = series(style, "sc16"), series(style, "greedy")
        width = 1.5 if style == "instruct" else 2
        ax.plot([r["gen_tokens"] for r in votes], [100 * r["acc"] for r in votes], color=color, lw=width,
                marker="o", ms=5.5, mec=SURFACE, mew=1, zorder=3, solid_capstyle="round")
        ax.plot([r["gen_tokens"] for r in greedy], [100 * r["acc"] for r in greedy], ls="none", marker="D",
                ms=6.5, mfc=SURFACE, mec=color, mew=1.75, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(3, 60000)
    ax.set_xticks([10, 100, 1000, 10000], ["10", "100", "1,000", "10,000"])
    ax.set_ylim(0, 100)
    ax.set_title(SIZES[size], loc="left", fontsize=10, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Generated tokens per question (log scale)", fontsize=9, color=MUTED)
    ax.annotate(f"{BUDGET} tokens", (BUDGET, 3), xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    rows = load_points(args.folder)
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True, facecolor=SURFACE)
    for ax, size in zip(axes, SIZES):
        draw(ax, rows, size)
    axes[0].set_ylabel("Accuracy on gsm8k_val (%)", fontsize=9, color=MUTED)

    handles = [Line2D([], [], color=color, lw=2, marker="o", ms=5.5, mec=SURFACE, label=label)
               for label, color in [*STYLES.values(), INSTRUCT]]
    handles.append(Line2D([], [], ls="none", marker="D", ms=6, mfc=SURFACE, mec=MUTED, mew=1.5, label="One greedy answer"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8, labelcolor=INK,
               handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    fig.savefig(args.out.with_suffix(".pdf"))
    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {args.out}.png, {args.out}.pdf and {len(rows)} points")


if __name__ == "__main__":
    main()
