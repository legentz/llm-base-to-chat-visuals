# RLVR, GRPO and reasoning RL

Research notes, verified 2026-11-09 against arXiv full text / official pages. Items not checked against a primary source are marked UNVERIFIED.

## 1. RLVR: the term

- "Reinforcement Learning with Verifiable Rewards (RLVR)" is named as "a novel method" in Tulu 3 (Lambert et al., AI2, arXiv 2411.15124, Nov 2024), the third stage after SFT and DPO, on Llama 3.1 bases.
- Idea: the reward is not a learned preference model but a programmatic check of the output. Reward is 1 (or a constant) if the verifier passes, else 0.
- Verifiers in practice:
  - Math: extract the final answer (e.g. `\boxed{}`) and compare to ground truth (string or symbolic equivalence). DAPO uses an `is_equivalent(a, o)` check.
  - Code: run the program against hidden unit tests (DeepSeek-R1 paper: compiler plus predefined test cases).
  - Format: reward for wrapping reasoning in `<think>...</think>`. R1-Zero: Reward = Reward_acc + Reward_format, equal weight.
  - Instruction following: rule-checkable constraints (Olmo 3 includes an instruction-following domain). Judge-LLM or reward-model rewards cover the rest (Qwen3 uses model-based rewards with and without a reference answer).
- Failure modes: false negatives from strict string matching, reward hacking (MiMo-V2-Flash has an appendix titled "Reward Hacking of SWE-Bench"), and no verifier at all for open-ended tasks.

## 2. GRPO (DeepSeekMath, Shao et al. 2024, arXiv 2402.03300)

Critic-free PPO variant. For each prompt $q$ sample a group $\{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}$, score with rewards $r_i$, and use the group as the baseline:

$$\hat A_{i,t}=\frac{r_i-\mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\})}$$

$$\mathcal J_{GRPO}=\mathbb E\Big[\tfrac1G\sum_{i=1}^G\tfrac1{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)-\beta\,D_{KL}(\pi_\theta\|\pi_{ref})\Big)\Big],\quad \rho_{i,t}=\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}$$

- KL is the per-token "k3" estimator (Schulman), always $\ge 0$: $\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$. It is added to the loss, not to the reward (unlike PPO).
- No value network, so memory is lower than PPO; the paper's abstract states this.
- DeepSeekMath RL config: lr 1e-6, KL coef 0.04, 64 outputs per question, max length 1024, batch 1024.
- Toy: G=4, rewards [1,0,0,1] gives mean 0.5, std 0.5, A = [+1,-1,-1,+1]. Rewards [1,0,0,0] give mean 0.25, std 0.433, A = [+1.73,-0.58,-0.58,-0.58]. All-equal rewards (all 0 or all 1) give A=0 and therefore no gradient.

## 3. DeepSeek-R1-Zero and R1 (arXiv 2501.12948; v2 Jan 2026 is the Nature-style version)

- R1-Zero: RL directly on DeepSeek-V3-Base with GRPO and rule rewards only (accuracy + format). No SFT.
- v2 hyperparameters: lr 3e-6, KL coef 0.001, temperature 1, 16 samples per question, max length 32,768 tokens until step 8.2k and 65,536 after, 10,400 steps (about 1.6 epochs), 32 questions per step (batch 512). Reference model is replaced with the latest policy every 400 steps.
- AIME 2024 pass@1 rises from 15.6% to 77.9% over training (v2 text). I did not verify the older v1 figure.
- Response length grows steadily ("learns to solve reasoning tasks with more thinking time", Fig. 1b). The v2 text says length and accuracy both jump at step 8.2k, when the cap is raised. Exact token counts are only in the figure: UNVERIFIED.
- "Aha moment" (Table 2): an intermediate checkpoint writes "Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step..." It is a self-reflection, backtracking behaviour that emerged without being rewarded directly.
- Caveat: Dr. GRPO paper finds DeepSeek-V3-Base already shows an "Aha moment" before RL, and that Qwen2.5 bases have strong pre-existing reasoning.
- R1 pipeline (Fig. 2): cold-start SFT, reasoning RL (adds a language-consistency reward), rejection-sampling SFT, then RL for helpfulness/harmlessness. v2 data: about 600k reasoning samples (correct only, mixed-language and code-block CoT filtered out) plus about 200k non-reasoning samples. The 800k set is the distillation set. Intermediate checkpoints are named Dev1, Dev2, Dev3.
- Dev1 (after cold start) lost some AIME score versus R1-Zero but gained IF-Eval and ArenaHard.
- Distillation: see distillation.md. Six distilled checkpoints are listed in Table 6 (1.5B to 70B).

## 4. Variants

### DAPO (Yu et al., ByteDance Seed / Tsinghua AIR, arXiv 2503.14476)

Qwen2.5-32B base reaches 50 on AIME 2024 (avg@32), versus 47 for DeepSeek-R1-Zero-Qwen-32B, in about 50% of the steps. Four techniques:

1. Clip-Higher: decoupled range $[1-\varepsilon_{low},1+\varepsilon_{high}]$, with $\varepsilon_{low}=0.2$, $\varepsilon_{high}=0.28$. This lets low-probability tokens grow and avoids entropy collapse.
2. Dynamic sampling: keep only prompts with $0<\#\text{correct}<G$ (constraint on the objective), over-sampling and filtering to keep non-zero-advantage groups.
3. Token-level loss: normalise by total tokens in the batch, $\frac{1}{\sum_i|o_i|}\sum_i\sum_t$, instead of per-sample $\frac1{|o_i|}$. Adds stability and healthier length growth.
4. Overlong reward shaping: a soft penalty $R_{len}(y)=0$ if $|y|\le L_{max}-L_{cache}$; $\frac{(L_{max}-L_{cache})-|y|}{L_{cache}}$ in the soft interval; $-1$ if $|y|>L_{max}$. Settings: 16,384 expected max plus 4,096 cache (20,480 hard cap).

- Setup: batch 512 prompts, 16 samples each, lr 1e-6. The KL term is dropped (standard in later work).
- Ablation (AIME24 avg@32): naive GRPO 30, +overlong filtering 36, +clip-higher 38, +soft overlong 41, +token-level loss 42, +dynamic sampling 50.

### Dr. GRPO (Liu et al., Sea AI Lab / NUS, arXiv 2503.20783, "Understanding R1-Zero-Like Training")

- Identifies two biases in the GRPO objective. The $1/|o_i|$ factor favours long wrong answers (a negative advantage is diluted over more tokens, so long errors are penalised less) and short right ones. The std division up-weights very easy and very hard questions.
- Fix: delete both. $\hat A_{i}=r_i-\mathrm{mean}(\{r_j\})$ and sum tokens with a constant normaliser.
- Reported: 43.3% on AIME 2024 with a 7B base and a minimalist recipe, with better token efficiency (shorter wrong answers).

### GSPO (Zheng et al., Qwen, arXiv 2507.18071)

- Importance ratio on the whole sequence, length-normalised: $s_i(\theta)=\Big(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\Big)^{1/|y_i|}=\exp\Big(\frac1{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{\theta_{old}}(y_{i,t}|\cdot)}\Big)$.
- Clip whole responses: $\min(s_i\hat A_i,\ \mathrm{clip}(s_i,1-\varepsilon,1+\varepsilon)\hat A_i)$, with group-normalised $\hat A_i$.
- Clip ranges are orders of magnitude smaller than GRPO's: 3e-4 and 4e-4 versus GRPO's tuned 0.2 and 0.27 in the paper's experiment.
- Motivation: token-level ratios are noisy, and for MoE models the experts routed change between old and new policy. GRPO needed a "Routing Replay" trick to converge, and GSPO does not. The abstract says it contributed to the Qwen3 improvements.

### Others (briefly)

- CISPO (MiniMax-M1, arXiv 2506.13585): clips importance-sampling weights rather than token updates. Full RL ran in about 3 weeks on 512 H800 GPUs.
- Yue et al. 2025 (arXiv 2504.13837): at large pass@k the base model beats the RLVR model, which suggests RLVR sharpens rather than adds capacity, while distillation can add new patterns.

## 5. Production recipes

- Qwen3 (arXiv 2505.09388): 4 stages: long-CoT cold start, reasoning RL, thinking-mode fusion, general RL. Flagship models run all four; small models (0.6B to 14B, 30B-A3B) use strong-to-weak distillation. Reasoning RL uses GRPO on 3,995 query-verifier pairs that were not used in the cold start, are learnable and as hard as possible, and cover diverse sub-domains. Large batch and many rollouts per query are used.
- Olmo 3 (AI2, arXiv 2512.13961): Think = SFT (Dolci) then DPO (delta learning) then RLVR (OlmoRL). Instruct follows a parallel path, and RL-Zero runs RLVR from the base model. OlmoRL builds on GRPO plus DAPO and Dr. GRPO ideas: token-level loss, no KL loss, clip-higher, a more efficient dynamic sampling. About 100K prompts in Dolci-Think-RL across math, code, instruction following and chat; 4x faster RL training. RL gains are larger after DPO than directly after SFT (Fig. 19). Olmo 3.1 Think 32B shows extended RL helps further.
- 2026 (limited verification; arXiv abstracts only):
  - REAL, "Rewards as Labels" (arXiv 2602.05630, Feb 2026): treats verifiable rewards as classification labels, and reports +6.7 Pass@1 over DAPO at 1.5B.
  - ConSPO (arXiv 2605.12969, May 2026): contrastive InfoNCE-style objective over length-normalised sequence log-likelihoods.
  - MiMo-V2-Flash (arXiv 2601.02780): scaled agentic RL (about 120K environments) and then multi-teacher on-policy distillation, see distillation.md.
  - No single successor to GRPO has emerged; most 2026 work edits the ratio, advantage or credit assignment. Other 2026 venues (ICML/ICLR lists) are UNVERIFIED.

## Suggested visuals

1. Group advantage widget: slider for G=8 rewards; show bars of $r_i$, mean line, and $A_i$ with and without std (Dr. GRPO). Example rewards [1,1,0,0,0,0,0,0] give A=+1.89 for correct and -0.63 for wrong (population std); without std, +0.75 and -0.25.
2. Clip-region plot: ratio on x (0.5 to 1.5), objective on y, for $A>0$ and $A<0$, comparing symmetric $\varepsilon=0.2$ with DAPO's 0.2/0.28.
3. Line chart of R1-Zero AIME pass@1 15.6 to 77.9 over about 10k steps next to mean response length (qualitative, label axes "illustrative" since only endpoints verified).
4. DAPO ablation waterfall: 30, 36, 38, 41, 42, 50.
5. Token vs sequence ratio: 3 tokens with ratios 1.3, 0.9, 1.1, giving geometric mean $s=(1.3\cdot0.9\cdot1.1)^{1/3}=1.09$.
6. Length-bias demo: wrong answer of 100 vs 1000 tokens; per-token weight with $1/|o|$ (0.01 vs 0.001) versus constant.

## References

1. Lambert et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." 2024. https://arxiv.org/abs/2411.15124
2. Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024. https://arxiv.org/abs/2402.03300
3. DeepSeek-AI (Guo et al.). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025 (v2 Jan 2026). https://arxiv.org/abs/2501.12948
4. Yu, Zhang, Zhu et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." 2025. https://arxiv.org/abs/2503.14476
5. Liu, Chen, Li, Qi, Pang, Du, Lee, Lin. "Understanding R1-Zero-Like Training: A Critical Perspective." 2025. https://arxiv.org/abs/2503.20783
6. Zheng, Liu, Li, Chen, Yu, Gao, Dang, Liu, Men, Yang, Zhou, Lin. "Group Sequence Policy Optimization." 2025. https://arxiv.org/abs/2507.18071
7. MiniMax (Chen et al.). "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention." 2025. https://arxiv.org/abs/2506.13585
8. Yue, Chen, Lu, Zhao, Wang, Yue, Song, Huang. "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" 2025. https://arxiv.org/abs/2504.13837
9. Qwen Team. "Qwen3 Technical Report." 2025. https://arxiv.org/abs/2505.09388
10. Olmo Team (AI2). "Olmo 3." 2025. https://arxiv.org/abs/2512.13961
11. Zhai, Chen, Zhao, Qian, Shen, Lu. "Rewards as Labels: Revisiting RLVR from a Classification Perspective." 2026. https://arxiv.org/abs/2602.05630
12. Zhang, Ma, Dong, Leng, Zhao, Sun, Yang, Jiang. "Revisiting Reinforcement Learning with Verifiable Rewards from a Contrastive Perspective." 2026. https://arxiv.org/abs/2605.12969
13. LLM-Core Xiaomi. "MiMo-V2-Flash Technical Report." 2026. https://arxiv.org/abs/2601.02780
