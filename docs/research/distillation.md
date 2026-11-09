# Distillation: from logit matching to on-policy, multi-teacher

Research notes, verified 2026-11-09 against arXiv full text and official pages. UNVERIFIED marks unchecked items.

## 1. Logit knowledge distillation (Hinton, Vinyals, Dean 2015, arXiv 1503.02531)

- Raise the softmax temperature $T$ so the teacher gives "suitably soft" targets, and train the student at the same $T$. Matching logits is a special case.
- Soft distribution: $p_i^{(T)}=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$.
- Loss: $\mathcal L=(1-\alpha)\,\mathrm{CE}(y,\sigma(z_s))+\alpha\,T^2\,\mathrm{KL}\big(p_T^{teacher}\,\|\,p_T^{student}\big)$.
- Soft-target gradients scale as $1/T^2$, so the paper multiplies them by $T^2$ when mixing with hard-label loss. It also found a considerably lower weight on the hard-target term worked best.
- KL direction: teacher first, so this is forward KL: $\mathrm{KL}(P\|Q)=\sum_c P(c)\log\frac{P(c)}{Q(c)}$. Minimising it is cross-entropy against the teacher distribution (mass-covering).
- LLM version: apply per token over the vocabulary of the teacher's context (needs a shared tokenizer).

## 2. Sequence-level KD (Kim & Rush 2016, arXiv 1606.07947)

- Word-level KD matches the teacher's per-position distributions on gold sequences. Sequence-level KD trains the student on the teacher's beam-search output (the highest-scoring sequence), which approximates the teacher's sequence distribution by its mode.
- Third variant, sequence-level interpolation: pick the beam candidate most similar to the gold target.
- Result (NMT): a 2x500 LSTM student roughly matches a 4x1000 LSTM teacher.
- This is the ancestor of modern "SFT on teacher outputs", which is the dominant LLM recipe.

## 3. Distilling reasoning traces by SFT (DeepSeek-R1, arXiv 2501.12948)

- About 800k curated samples from R1 (about 600k reasoning plus about 200k non-reasoning), used for plain SFT with no RL stage on the students: 2-3 epochs, max length 32,768, batch size 64, cosine decay to 1/10 of the initial lr.
- Six released students (Table 6, base model and initial lr):

| Student              | Base                   | lr   |
| -------------------- | ---------------------- | ---- |
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B      | 1e-4 |
| R1-Distill-Qwen-7B   | Qwen2.5-Math-7B        | 8e-5 |
| R1-Distill-Qwen-14B  | Qwen2.5-14B            | 7e-5 |
| R1-Distill-Qwen-32B  | Qwen2.5-32B            | 6e-5 |
| R1-Distill-Llama-8B  | Llama-3.1-8B           | 5e-5 |
| R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 2e-5 |

- Later example: DeepSeek-R1-0528-Qwen3-8B. The HF card says the chain-of-thought of R1-0528 was distilled to post-train Qwen3 8B Base, +10.0% on AIME 2024 over Qwen3 8B, matching Qwen3-235B-thinking (card claim).
- Note: this is hard-label (sampled token) distillation, not logit matching. Student never sees teacher probabilities. See rlvr.md for Yue et al., who report that distillation can add reasoning patterns that RLVR alone does not.

## 4. On-policy distillation (GKD, Agarwal et al. 2024, arXiv 2306.13649, ICLR 2024)

- Problem: SFT-style distillation trains on a fixed set of teacher outputs, but at inference the student conditions on its own (possibly flawed) prefixes (train/inference mismatch).
- GKD: sample sequences from the student, and use the teacher's token probabilities as labels on those sequences. Divergence is a free choice:
  - forward KL (mean-seeking), reverse KL (mode-seeking), or generalized JSD (interpolates).
  - Paper states forward KL under an empirical distribution equals maximum likelihood, and that with capacity mismatch reverse KL is mode-seeking and forward KL mean-seeking.
- Generalised objective with student sampling fraction $\lambda$: $\mathcal L(\theta)=(1-\lambda)\,\mathbb E_{(x,y)\sim D}[D(P_T\|P_S^\theta)(y|x)]+\lambda\,\mathbb E_{x}\,\mathbb E_{y\sim P_S(\cdot|x)}[D(P_T\|P_S^\theta)(y|x)]$. I wrote this from the paper's description (supervised-data term mixed with an on-policy term), so check exact notation before publishing: UNVERIFIED notation.
- Covers summarisation, translation, arithmetic reasoning with T5 students.

## 5. Thinking Machines "On-Policy Distillation" (Kevin Lu et al., 27 Oct 2025)

- Student samples rollouts; frozen teacher scores each token; reward at each token is the negative per-token reverse KL, with discount 0.
- $$\mathrm{KL}(\pi_\theta\|\pi_{T})=\mathbb E_{x\sim\pi_\theta}\big[\log\pi_\theta(x_{t+1}|x_{1..t})-\log\pi_T(x_{t+1}|x_{1..t})\big],\qquad A_t=-\big(\log\pi_\theta-\log\pi_T\big)$$
- Positioning table: SFT = off-policy and dense; RL = on-policy and sparse; on-policy distillation = on-policy and dense.
- Numbers from the post (Qwen3-8B-Base student, Qwen3-32B teacher): from a 400K-prompt SFT checkpoint at 60% AIME'24, on-policy distillation reaches 70% in about 150 steps (about 77K prompts, 4 samples each). Estimated 9-30x less FLOPs than extending SFT to 2M prompts, depending on whether teacher sampling cost is counted.
- Personalisation case: mid-training on internal documents dropped IF-eval 85% to 45%, and distilling from the earlier Qwen3-8B recovered 83% while keeping the knowledge gain (internal QA 41%).
- Reverse KL is mode-seeking (GKD paper) and can be estimated from the single sampled token at each position. The claim that it is hard to reward-hack is from memory of the post: UNVERIFIED.

## 6. Qwen3 strong-to-weak distillation (arXiv 2505.09388)

- Applies to 0.6B, 1.7B, 4B, 8B, 14B dense and Qwen3-30B-A3B. Flagships (235B-A22B, 32B) use the 4-stage pipeline.
- Phase 1, off-policy: teacher outputs in both /think and /no_think modes are used for response distillation, to give basic reasoning and mode switching.
- Phase 2, on-policy: the student generates in either mode, then is fine-tuned by aligning its logits with a teacher (Qwen3-32B or Qwen3-235B-A22B) to minimise KL.
- Claims: better Pass@1 and Pass@64 than RL, with 1/10 of the GPU hours of the four-stage recipe.
- Table 21 (Qwen3-8B, starting from the same off-policy distilled checkpoint):

| Method                   | AIME'24 (pass@64) | AIME'25 (pass@64) | GPU hours |
| ------------------------ | ----------------- | ----------------- | --------- |
| Off-policy distillation  | 55.0 (90.0)       | 42.8 (83.3)       | n/a       |
| + RL                     | 67.6 (90.0)       | 55.5 (83.3)       | 17,920    |
| + On-policy distillation | 74.4 (93.3)       | 65.5 (86.7)       | 1,800     |

- RL did not improve pass@64; on-policy distillation did (AIME'24 90.0 to 93.3).

## 7. Multi-Teacher On-Policy Distillation (MOPD), MiMo-V2-Flash (Xiaomi, arXiv 2601.02780, Jan 2026)

- Verified in the paper. Model: 309B MoE, 15B active, pre-trained on 27T tokens.
- Domain-specialist teachers (e.g. trained by large-scale RL) provide dense token-level rewards, and the student learns on its own samples. Teacher is chosen by prompt domain $\pi_{domain_x}$.
- Reverse-KL loss: $\mathcal L=-\mathbb E_{x\sim D,\,y_t\sim\pi_\theta}\log\frac{\pi_{domain_x}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}$.
- Implemented as RL with per-token advantage $\hat A_{MOPD,t}=\mathrm{sg}\big[\log\frac{\pi_{domain_x}}{\pi_\theta}\big]$, training-inference importance weight $w_t$ with tokens outside $[\epsilon_{low},\epsilon_{high}]$ masked (weight 0). The text adds that outcome-reward advantages are combined by default (details not read).
- Pipeline: SFT, then domain-specific RL to train teachers, then MOPD merges them into one student. Purpose: avoid capability imbalance (one skill regressing as another improves).
- Other claims (card-level): comparable to DeepSeek-V3.2 and Kimi-K2 with about 1/2 and 1/3 of their total parameters.

## Suggested visuals

1. Temperature slider: logits [5.0, 3.0, 1.0, 0.5], T=1 gives about [0.86, 0.12, 0.016, 0.010]; T=4 gives about [0.43, 0.26, 0.16, 0.14]. Show dark knowledge appearing in the tail.
2. Forward vs reverse KL: bimodal teacher (0.45, 0.10, 0.45 over three tokens) and a unimodal student; show forward KL spreading mass and reverse KL picking one mode.
3. Off-policy vs on-policy diagram: student's own prefix (arrow) versus teacher's prefix; color tokens by per-token reverse KL (a token with $\log\pi_\theta-\log\pi_T=+2.1$ gets negative advantage).
4. Bar chart of Qwen3-8B Table 21: GPU hours 17,920 vs 1,800 against AIME'24 67.6 vs 74.4.
5. Pipeline strip: R1 -> 800k samples -> SFT -> six students; versus MOPD with N teachers feeding one student.

## References

1. Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." 2015. https://arxiv.org/abs/1503.02531
2. Kim, Rush. "Sequence-Level Knowledge Distillation." EMNLP 2016. https://arxiv.org/abs/1606.07947
3. Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist, Bachem. "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes." ICLR 2024. https://arxiv.org/abs/2306.13649
4. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025. https://arxiv.org/abs/2501.12948
5. DeepSeek-AI. Model card, DeepSeek-R1-0528-Qwen3-8B. 2025. https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
6. Lu, Kevin, et al. (Thinking Machines Lab). "On-Policy Distillation." 27 Oct 2025. https://thinkingmachines.ai/blog/on-policy-distillation/
7. Qwen Team. "Qwen3 Technical Report." 2025. https://arxiv.org/abs/2505.09388
8. LLM-Core Xiaomi. "MiMo-V2-Flash Technical Report." 2026. https://arxiv.org/abs/2601.02780
