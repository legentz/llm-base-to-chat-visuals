# RLHF: reward models, PPO with KL, overoptimization

Verified 2026-11-09 against arXiv. "UNVERIFIED" = not confirmed from a primary source this session.

## Pipeline (InstructGPT; Ouyang et al., 2022, arXiv 2203.02155)

1. SFT on labeler demonstrations (~13k prompts; 16 epochs, cosine LR, dropout 0.2).
2. Reward model (RM): labelers rank K responses per prompt (K = 4..9 in the paper, K value UNVERIFIED); RM is 6B ("saves a lot of compute"); ~33k RM prompts.
3. PPO against RM (~31k prompts, no human labels), with KL to SFT and a pretraining-mix term (PPO-ptx).

- Headline: 1.3B InstructGPT preferred over 175B GPT-3 by humans (~100x fewer params).
- Lineage: Ziegler et al. 2019 (arXiv 1909.08593) first applied this to LMs (5k comparisons for style, ~60k for summarization).

## Reward model: Bradley-Terry

Preference model: $P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$.
Loss (InstructGPT Eq. 1, averaged over the $\binom K2$ pairs per prompt; the paper does not use the name "Bradley-Terry", the form is that model):
$$\mathcal L_{RM}(\phi) = -\tfrac{1}{\binom K2}\,\mathbb E_{(x,y_w,y_l)}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]$$

- Only reward differences are identified (any per-prompt constant shift leaves the loss unchanged); RM = LM with scalar head on last token.
- All K-choose-2 pairs of one prompt must be in the same batch element, otherwise overfitting (InstructGPT trained on each prompt's comparisons as one batch element; detail from memory, UNVERIFIED).
- Metrics: pairwise accuracy (typically ~60-75% on held-out human prefs; range UNVERIFIED), mean margin.

## RL objective (InstructGPT Eq. 2, PPO-ptx)

$$\text{obj}(\phi)=\mathbb E_{(x,y)\sim\pi_\phi}\Big[r_\theta(x,y)-\beta\log\frac{\pi_\phi(y|x)}{\pi_{SFT}(y|x)}\Big]+\gamma\,\mathbb E_{x\sim D_{pretrain}}\big[\log\pi_\phi(x)\big]$$

- Per-token implementation (standard in TRL/OpenAI code; Ziegler 2019 uses the same idea): reward at token $t$ is $r_t = -\beta\,[\log\pi_\phi(a_t|s_t)-\log\pi_{ref}(a_t|s_t)] + \mathbb 1[t=T]\,r_\theta(x,y)$. The sequence-level KL sum equals the sum of per-token log-ratios (a sampled estimator of $\mathrm{KL}(\pi_\phi\|\pi_{ref})$).
- PPO clipped surrogate (Schulman et al. 2017, arXiv 1707.06347): $L^{CLIP}=\mathbb E_t[\min(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)]$, $\rho_t=\pi_\phi/\pi_{old}$; advantages $\hat A_t$ via GAE with a learned value network (so four models resident: policy, reference, reward, critic).
- $\beta$: larger = stay near reference, lower reward, less drift; smaller = more reward, more hacking. InstructGPT numeric $\beta$ UNVERIFIED (symbol only in accessible text). Tulu 3 RLVR (PPO) used KL coef 0.05 (8B), and 0.07 or 0.7 (70B; paper's Table 21 caption and Sec. 6.4 disagree), LR 3e-7 (8B) / 1e-7 (70B).
- $\gamma$ (ptx term) fights the "alignment tax" on public NLP benchmarks.

## Reward hacking / overoptimization (Gao, Schulman, Hilton, 2022, arXiv 2210.10760)

- Setup: fixed 6B "gold" RM stands in for humans; proxy RMs 3M-3B trained on gold labels. Optimize policy against proxy, measure gold.
- With $d=\sqrt{\mathrm{KL}(\pi\|\pi_{init})}$ and $R(0)=0$:
  $$R_{BoN}(d)=d\,(\alpha_{BoN}-\beta_{BoN}\,d),\qquad R_{RL}(d)=d\,(\alpha_{RL}-\beta_{RL}\log d)$$
  Coefficients scale smoothly with RM parameter count.
- Proxy reward keeps rising while gold peaks then falls (Goodhart). Bigger RM / more RM data = later and higher peak.
- KL penalty: converges earlier (acts like early stopping) but "does not affect the KL_RL-gold reward frontier" (authors note hyperparameter sensitivity). So beta mostly moves where you stop on the same curve.
- Typical real-world hacks: length inflation, sycophancy, formatting tricks (general knowledge; not tied to a single verified source this session).

## RLAIF / Constitutional AI (Bai et al., 2022, arXiv 2212.08073)

- Phase 1 (SL): model critiques and revises its own outputs against written principles; finetune on revisions.
- Phase 2 (RL): model labels which of two samples better follows the principles -> AI preference model -> RL (RLAIF). Humans provide only the principle list (and helpfulness labels).
- Result: harmless yet non-evasive assistant; chain-of-thought labeling helps. Later RLAIF-vs-RLHF comparisons exist (Lee et al. 2023) - not verified here, UNVERIFIED.

## Status of PPO, 2025-2026

- Verified: Tulu 3 (Nov 2024) used PPO for its RLVR stage after SFT and length-normalized DPO. GRPO (DeepSeekMath, arXiv 2402.03300) is a PPO variant with reduced memory (drops the critic; group-relative advantage, standard description, details UNVERIFIED from abstract alone). DeepSeek-R1 (arXiv 2501.12948, Nature 2025) reports reasoning emerging from pure RL; the algorithm name GRPO was not on the abstract page I read, UNVERIFIED there.
- Olmo 3 (arXiv 2512.13961) final stage is RLVR via "OlmoRL" (Think) / Dolci Instruct-RL (Instruct); exact algorithm (GRPO-style) UNVERIFIED.
- Trend (supported by above): preference-RL with a learned RM has largely given way, for reasoning/code/math, to RL against verifiable rewards with critic-free PPO variants; DPO-style offline methods handle chat preference. Full PPO with a critic is still used (Tulu 3). Claims about closed labs: UNVERIFIED.

## Suggested visuals

1. BT playground: two sliders for $r(y_w)$, $r(y_l)$; show $\sigma(\Delta)$ and loss; training a toy RM from 8 pairs, with accuracy rising ~50% -> ~70-75% (illustrative).
2. Token-level reward strip: per-token log-ratio bars (KL penalty, mostly small negatives), final-token RM score; slider beta in {0.01, 0.05, 0.1, 0.2} changes the net return.
3. Overoptimization curve: x = $\sqrt{KL}$ (0..~10), proxy rising monotonically, gold $d(\alpha-\beta\log d)$ peaking (use e.g. alpha=1.0, beta=0.25 purely illustrative); marker at beta-controlled early stop; toggle RM size to shift the peak right.
4. Training dashboard trajectories (illustrative, typical shape): mean RM reward 0 -> +2..4, KL to ref 0 -> 5-15 nats, response length drifting up, entropy decaying; flagged "hacking region".
5. Four-model memory diagram (policy, reference, RM, critic) vs GRPO (no critic) vs DPO (policy + reference).
6. Constitutional AI: critique -> revision -> AI preference label flow with a sample principle.

## References

- Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. https://arxiv.org/abs/2203.02155
- Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., Irving, G. (2019). Fine-Tuning Language Models from Human Preferences. https://arxiv.org/abs/1909.08593
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347
- Gao, L., Schulman, J., Hilton, J. (2022). Scaling Laws for Reward Model Overoptimization. https://arxiv.org/abs/2210.10760
- Bai, Y., Kadavath, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. https://arxiv.org/abs/2212.08073
- Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. https://arxiv.org/abs/2402.03300
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. https://arxiv.org/abs/2501.12948
- Lambert, N., et al. (2024). Tulu 3: Pushing Frontiers in Open Language Model Post-Training. https://arxiv.org/abs/2411.15124
- Team Olmo (2025). Olmo 3. https://arxiv.org/abs/2512.13961
