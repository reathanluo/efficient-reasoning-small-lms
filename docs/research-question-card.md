# Research Question Card — Topic G: Efficient Reasoning with Small LMs

_Drafted 2026-10-16. Status: tested on GSM8K test, 4 Nov 2026. H1-H3 are all not supported; see [Results](#results-on-gsm8k-test-4-nov-2026)._

## Research Question Card

**Question:**
For 0.5B–1.5B language models solving grade-school maths, where is a fixed token budget best spent? The options are:
- in training, by distilling rationales of different lengths (long chain-of-thought, standard, or concise);
- at inference, by self-consistency (sampling k answers and taking the majority vote).

A second question is whether the two strategies compose.

**Type:** confirmatory, with an exploratory error analysis.

**Hypotheses:**
- **H1:** At low and medium per-question token budgets, LoRA-SFT on concise rationales dominates the accuracy-vs-tokens Pareto front.
- **H2:** Concise SFT reduces sample diversity. As a result, accuracy gains per extra self-consistency sample are smaller than for standard-CoT SFT, so the two strategies compose sub-additively.
- **H3:** At ≤1.5B parameters, long-CoT SFT underperforms concise SFT, and it fails more often through non-termination or repetition. This is consistent with the "small model learnability gap" (ER-01).

**Why it matters:**
Small models run on cheap hardware and on devices, where inference tokens are the real cost. Prior work studies compression (ER-02, ER-04, ER-05) and test-time compute (ER-03) mostly separately, and mostly on larger models. Our question is how they trade off and interact at ≤1.5B under one accounting of compute.

**Current evidence:**
- ER-01 to ER-06, below. All come from abstracts and search summaries; we have not yet read the full papers.

**Missing evidence:**
- A matched comparison where the same problems are distilled at different rationale lengths into a 0.5B or 1.5B student, evaluated against self-consistency at an equal total token budget.
- The effect of rationale length on sample diversity.
- An out-of-distribution (OOD) check on SVAMP and on GSM-Symbolic-style perturbations, to probe both robustness and contamination.

**What would support it:**
- H1: the concise-SFT curve lies above the other curves at budgets ≤ ~512 tokens per question. This must hold on GSM8K test across 3 seeds, with a paired bootstrap 95% CI that excludes 0.
- H2: the number of unique answers among k=8 samples is lower for concise SFT, and the SC gain from k=1 to k=8 is smaller.
- H3: long-CoT SFT scores lower accuracy than concise SFT and has a higher truncation/repetition rate.

**What would falsify it:**
- H1 fails if the standard or long-CoT SFT curve matches or beats concise SFT at equal tokens, or if base/instruct + self-consistency dominates every SFT variant.
- H2 fails if the diversity metrics are equal, or if concise SFT gains as much or more from self-consistency.
- H3 fails if long-CoT SFT is at least as accurate as concise SFT at 1.5B.

Negative results are acceptable per the handout and should be reported.

### Results on GSM8K test (4 Nov 2026)

These are means over 3 seeds with paired-bootstrap 95% CIs, from `python -m reasoning.eval.hypotheses results/scored/gsm8k_test` (output in `results/tables/hypotheses_gsm8k_test.txt`). `gsm8k_val` gave the same verdicts.

- **H1: not supported.** Within 512 tokens, concise with voting loses to standard with one greedy answer, which uses no more tokens. It loses by 17.4 points [15.2, 19.6] at 0.5B (maj@4, 311 tokens, against 315) and 16.8 points [14.3, 19.2] at 1.5B (maj@8, 471 tokens, against 277). This holds on every seed. Standard SFT is the best curve at every budget, but only 0.7-1.8 points above the Instruct baseline, a gap we have not tested.
- **H2: not supported.** Concise answers are *more* diverse: among 8 samples they give 1.1 more unique answers at 0.5B and 1.6 more at 1.5B. The gain from k = 1 to 8 is smaller than standard's at 0.5B (8.9 against 13.4 points) but larger at 1.5B (11.3 against 9.2).
- **H3: not supported at 1.5B.** With one greedy answer, long beats concise by 5.6 points [3.0, 8.2] at 1.5B (McNemar p < 0.05 on every seed). At 0.5B they are tied (-0.4 points, p = 0.78). The failure-mode part holds: 32-34% of greedy long answers hit the 4,096-token cap, against 2-4% of concise ones. Long uses 10-17 times the tokens. At 1.5B, concise with 16 votes (61.2%, 941 tokens) beats long greedy (59.8%, 1,928 tokens).

**Minimal next action:**
1. Set up the environment on the NTU cluster: vLLM, transformers, peft, and the models downloaded.
2. Build a single evaluation harness with a tested answer parser.
3. Report Qwen2.5-0.5B and 1.5B baselines (base 8-shot CoT, and Instruct zero-shot) on a GSM8K validation split carved from the train set.

**Decision:** run experiment, after a team read of ER-01 and ER-02 to confirm positioning.

---

## Experimental design (draft)

**Students:** Qwen2.5-0.5B and Qwen2.5-1.5B base models, with LoRA. The Instruct versions serve as reference points only.

**Training data:** GSM8K train, 7,473 problems. Hold out ~500 problems as a validation set. **The GSM8K test set (1,319 problems) is touched only for the final runs.**

**Rationale conditions:** every condition uses the *same problems* and the *same count*, and keeps only traces whose answer is correct.

| ID | Condition | Source |
|---|---|---|
| L | Long CoT | A reasoning teacher, e.g. DeepSeek-R1-Distill-Qwen-7B |
| C | Concise | The **same trace as L**, rewritten or compressed by a teacher. Only the length/style changes, not the solution. |
| S | Standard CoT | Qwen2.5-7B-Instruct, normal prompt |
| A | Answer-only | Control with no rationale |

**Inference strategies:**
- greedy decoding;
- self-consistency with k ∈ {1, 2, 4, 8, 16};
- hard token caps of {64, 128, 256, 512, 1024}.

**Cost metrics:**
- primary: generated tokens per question;
- also: approximate FLOPs (≈ 2 · N_params · total tokens, including the prompt);
- also: vLLM wall-clock latency on the same GPU.

**Evaluation sets:**
- GSM8K test (in-distribution);
- SVAMP (OOD);
- GSM-Symbolic or GSM-Plus perturbations (robustness and contamination check);
- optionally MATH-500 (harder).

**Statistics:**
- 3 SFT seeds per condition;
- paired bootstrap or McNemar tests on per-question correctness;
- report 95% CIs.

**Analyses:**
- an error taxonomy over ~100 failures per condition. Categories: arithmetic slip, wrong plan, misread question, non-termination/repetition, truncation, formatting/parse;
- answer diversity under sampling.

## Evidence Records

| ID | Source | Source type | Relevance | Strength |
|---|---|---|---|---|
| ER-20261016-smallreason-01 | Li et al., "Small Models Struggle to Learn from Strong Reasoners", arXiv 2502.12143 (ICML 2025) | preprint (abstract only so far) | ≤3B students learn better from short CoT; proposes Mix Distillation. Closest prior work for H3. | observed |
| ER-20261016-smallreason-02 | Xia et al., "TokenSkip", arXiv 2502.12067 (EMNLP 2025) | preprint (abstract only) | Token-pruned CoT for SFT. A candidate method for building condition C. | observed |
| ER-20261016-smallreason-03 | Snell et al., "Scaling LLM Test-Time Compute Optimally…", arXiv 2408.03314 (ICLR 2025) | preprint (abstract only) | Framing for trading test-time against other compute. | observed |
| ER-20261016-smallreason-04 | Munkhbat et al., "Self-Training Elicits Concise Reasoning in LLMs" (ACL Findings 2025) | abstract only | ~30% fewer tokens on GSM8K/MATH at similar accuracy. | observed |
| ER-20261016-smallreason-05 | Jiang et al., "DRP: Distilled Reasoning Pruning…" (ACL Findings 2026) | abstract only | Teacher-pruned traces give a student shorter and more accurate outputs; "match CoT to student capacity". | observed |
| ER-20261016-smallreason-06 | Lee et al., "How Well do LLMs Compress Their Own Chain-of-Thought?", arXiv 2503.01141 | preprint (abstract only) | Prompted brevity gives modest compression. A baseline for "concise by prompting only". | observed |

All six records must be read in full before any claim enters the report. Verify each reference against the original paper; the handout deducts marks for hallucinated content.
