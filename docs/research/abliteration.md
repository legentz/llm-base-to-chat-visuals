# Abliteration: refusal as a direction in activation space

Scope: conceptual and evaluative. This file explains the geometry and the trade-offs. It deliberately contains no runnable code and no recipe for uncensoring a particular model. Verified 2026-11-09; UNVERIFIED marks unchecked items.

## 1. The finding (Arditi et al. 2024, arXiv 2406.11717, v3 Oct 2024)

- Claim: across 13 open-source chat models up to 72B parameters, refusal is mediated by a one-dimensional subspace of the residual stream. Erasing it stops refusals of harmful requests, and adding it makes the model refuse harmless ones.
- Authors: Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. The paper describes the result as a white-box jailbreak with minimal effect on other capabilities, and says it shows the brittleness of current safety fine-tuning.

## 2. Geometry

Take residual activations $x_i^{(l)}(t)$ at layer $l$ and post-instruction token position $i$ (the chat-template tokens after the user text).

- Difference in means (paper uses 128 harmful and 128 harmless training prompts):
  $$\mu_i^{(l)}=\frac1{|D_h|}\sum_{t\in D_h}x_i^{(l)}(t),\quad \nu_i^{(l)}=\frac1{|D_0|}\sum_{t\in D_0}x_i^{(l)}(t),\quad r_i^{(l)}=\mu_i^{(l)}-\nu_i^{(l)},\quad \hat r=r/\|r\|$$
- This gives $|I|\times L$ candidate vectors; one is selected on a validation set (32 harmful prompts from HarmBench validation, plus harmless ones) by three scores:
  - bypass score (refusal metric under ablation, lower is better);
  - induce score (must be $>0$ under activation addition);
  - KL score (average KL on harmless prompts at the last token, must be $<0.1$);
  - plus the constraint layer $l<0.8L$ so the direction is not just an unembedding direction for tokens like "I" or "As".
- Directional ablation (inference-time, applied at all layers and positions):
  $$x'=x-\hat r\hat r^{\top}x$$
- Activation addition (one layer; induces refusal on harmless input): $x'=x+r$.
- Weight orthogonalization (permanent, no runtime hook). For each matrix $W_{out}$ that writes to the residual stream (embedding, attention output, MLP output):
  $$W_{out}'=W_{out}-\hat r\hat r^{\top}W_{out}$$
  After this no component can write along $\hat r$, which is mathematically equivalent to directional ablation at every layer, and the model is unchanged otherwise. The paper presents it as simpler than fine-tuning-based jailbreaks.
- Intuition: picture the residual stream as a vector space where "harmful-looking" and "harmless-looking" prompts form two clouds. $r$ points from one cloud's centre to the other's. Projection removes the coordinate along that axis and leaves the other $d-1$ axes untouched.

## 3. Measured side effects (original paper)

- Quantified with LM Evaluation Harness: MMLU, ARC, GSM8K, TruthfulQA. MMLU, ARC and GSM8K are close to baseline (e.g. Llama-3 70B: MMLU 79.8 vs 79.9, GSM8K 90.8 vs 91.2). TruthfulQA drops consistently, by 1.0 to 3.5 points in the table's largest models (Gemma 7B -2.4, Yi 34B -3.5, Llama-3 70B -2.3).
- Except for Qwen 7B and Yi 34B, orthogonalised metrics lie within 99% confidence intervals of the original.
- Directional ablation is reported as more surgical than activation-addition-based methods.

## 4. Community path: FailSpy and Labonne

- FailSpy published the abliterator notebook and library, and a collection of abliterated models. Maxime Labonne's HF blog "Uncensor any LLM with abliteration" (13 June 2024) builds on that notebook.
- Labonne's blog reports a benchmark drop on the abliterated Daredevil-8B and uses DPO fine-tuning to "heal" it, giving NeuralDaredevil-8B-abliterated, which still lagged on GSM8K.
- The same author noted for Qwen3 that no single abliteration strategy fitted all sizes and that thinking mode interfered with non-reasoning refusals (from the mlabonne/Qwen3-8B-abliterated card).

## 5. Refinements (grimjim / Jim Lai, HF community blogs)

- Projected abliteration (25 Oct 2025): split $r=r_{\parallel}+r_{\perp}$ relative to the harmless mean $\hat\mu_0$, and ablate only the orthogonal part:
  $$r_{proj}=r-(r\cdot\hat\mu_0)\hat\mu_0$$
  Rationale: the parallel part pushes away from the "harmless/compliant" direction and has no principled reason to be removed. Reported on Gemma 3 12B it with no benchmarks in the post body.
- Norm-preserving biprojected abliteration (6 Nov 2025): factor each weight row into magnitude and direction, $W=M\hat W$; ablate only on the unit-norm directions $\hat W'=\hat W-\alpha\hat r(\hat r^{\top}\hat W)$, renormalise rows, and rebuild $W_{new}=M\,\hat W'/\|\hat W'\|$ so each row keeps its original L2 norm. "Biprojection" adds the harmless-overlap removal even when the direction is measured at another layer. Layer choice uses a heuristic: $\mathrm{SNR}\times(1-\cos)$.
- Author's own numbers on Gemma-3 12B-it (UGI leaderboard): baseline UGI 19.58 / NatInt 18.72; standard abliteration 32.08 / 18.64; norm-preserving biprojected 32.61 / 21.33. Single model, self-reported. The post itself calls the layer selection empirical and mentions a Hydra-like self-repair, so multi-layer intervention is needed.

## 6. Comparative analysis (Young, arXiv 2512.13655, Dec 2025, v2 Jan 2026)

- Evaluates four tools (Heretic, DECCP, ErisForge, FailSpy) on sixteen instruction-tuned 7B-14B models, with quantitative metrics only on subsets: Heretic KL and refusal rate on eight models, benchmarks on three plus Zephyr-7B-beta. Models include Qwen2.5-7B-Instruct, Qwen3-8B and Qwen3-14B.
- Single-pass methods preserved capability best (average GSM8K change: ErisForge -0.28 pp, DECCP -0.13 pp). Bayesian-optimised abliteration gave variable distribution shift (KL 0.043 to 1.646) with model-dependent impact.
- Principal finding: mathematical reasoning (GSM8K) is the most sensitive capability, with changes from +1.51 pp to -18.81 pp (-26.5% relative) depending on tool and model.
- Caveat: small samples, one author, tools differ in their evaluation sets.

## 7. Heretic (p-e-w, Philipp Emanuel Weidmann)

- Training-free tool, first released Nov 2025, AGPL-3.0. Wraps a modifier (directional ablation or an arbitrary-rank ablation) with an Optuna TPE optimiser that searches parameters (layer ranges, per-component weights, direction index) against scorers. Default scorers: refusal rate and KL divergence from the original model on harmless prompts.
- The trade-off is a Pareto front: fewer refusals usually cost more KL.
- Worked example from a published model card (Qwen3-4B-Instruct-2507, Heretic v1.0.0): KL divergence 0.43; refusals 21/100 versus 99/100 for the original. The card also lists per-component weight curves (`attn.o_proj`, `mlp.down_proj`) with a `direction_index` of 30.93, i.e. a fractional (interpolated) direction layer.
- README says it takes about 30-40 minutes on an RTX 3090 for Qwen3-4B-Instruct-2507.

## 8. Defences

- Extended-refusal fine-tuning (Abu Shairah et al., arXiv 2505.19056): train the model to refuse with a detailed justification, so the refusal signal is spread over many token positions and dimensions instead of one direction. Tested on Llama-2-7B-Chat and Qwen2.5-Instruct 1.5B/3B (CatQA benchmark, 10,000-example mix incl. 5,711 Alpaca-GPT4 pairs): refusal rates after abliteration fall by at most about 10 pp, versus 70-80 pp for baselines, with a small utility cost (MMLU -0.5 to -1.3). Small scale and preprint.
- Open weights give no technical defence beyond making the direction harder to isolate, and dispersal is not shown to hold against stronger attackers (UNVERIFIED beyond the paper).

## 9. Real abliterated Qwen repos (exist on HF, checked 2026-11-09 via the HF API)

| Repo id                                        | Base (from metadata)                                                     | Notes                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `mlabonne/Qwen3-8B-abliterated`                | `Qwen/Qwen3-8B` (finetune)                                               | tags `abliterated`; card calls it a research project, experimental |
| `huihui-ai/Huihui-Qwen3-8B-abliterated-v2`     | `Qwen/Qwen3-8B` (finetune)                                               | tags `abliterated`, `uncensored`; 20k downloads when checked       |
| `p-e-w/Qwen3-4B-Instruct-2507-heretic`         | base not declared in metadata (card links `Qwen/Qwen3-4B-Instruct-2507`) | Heretic v1.0.0; KL 0.43, refusals 21/100                           |
| `huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2` | `Qwen/Qwen2.5-7B-Instruct` (finetune)                                    | older (Sept 2024)                                                  |

## Suggested visuals

1. 2D scatter of activations: harmless cloud centred (-1, 0), harmful cloud centred (+1, 0), $r=(2,0)$. Slider moves points to $x-\hat r\hat r^\top x$ and collapses them onto the vertical axis.
2. Heatmap layers x positions (e.g. 32 layers x 5 post-instruction tokens) of bypass score, with the feasible region ($l<0.8L$, KL<0.1) outlined.
3. Pareto plot, refusals out of 100 (y) vs KL (x): original (0 KL, 99), Heretic-style point (0.43, 21), and a spread of trials from (0.05, 80) down to (1.5, 3). Label as illustrative except the (0.43, 21) point.
4. Capability bars from Arditi Table 3 (Llama-3 70B MMLU 79.8 vs 79.9; TruthfulQA 59.5 vs 61.8), contrasted with the worst GSM8K change in the comparative study (-18.81 pp).
5. Row-norm diagram for norm preservation: weight row length before/after, standard (shrunk) versus norm-preserving (same length).

## References

1. Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. "Refusal in Language Models Is Mediated by a Single Direction." 2024. https://arxiv.org/abs/2406.11717
2. Labonne. "Uncensor any LLM with abliteration." Hugging Face blog, 13 Jun 2024. https://huggingface.co/blog/mlabonne/abliteration
3. Lai (grimjim). "Projected Abliteration." HF blog, 25 Oct 2025. https://huggingface.co/blog/grimjim/projected-abliteration
4. Lai (grimjim). "Norm-Preserving Biprojected Abliteration." HF blog, 6 Nov 2025. https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
5. Young. "Comparative Analysis of LLM Abliteration Methods: A Cross-Architecture Evaluation." 2025. https://arxiv.org/abs/2512.13655
6. Weidmann (p-e-w). Heretic. https://github.com/p-e-w/heretic
7. Abu Shairah et al. "An Embarrassingly Simple Defense Against LLM Abliteration Attacks." 2025. https://arxiv.org/abs/2505.19056
8. Model cards: https://huggingface.co/mlabonne/Qwen3-8B-abliterated ; https://huggingface.co/huihui-ai/Huihui-Qwen3-8B-abliterated-v2 ; https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic
