# DPO and relatives: IPO, KTO, ORPO, SimPO

Verified 2026-11-09 against arXiv abs/html and the TRL docs. "UNVERIFIED" = not confirmed from a primary source this session.

## DPO (Rafailov, Sharma, Mitchell, Ermon, Manning, Finn; 2023, arXiv 2305.18290)

- Starts from the KL-constrained RLHF objective $\max_\pi \mathbb E[r(x,y)]-\beta\,\mathrm{KL}(\pi\|\pi_{ref})$, whose optimum is closed-form:
  $$\pi^*(y|x)=\tfrac1{Z(x)}\pi_{ref}(y|x)\exp\!\big(r(x,y)/\beta\big)\ \Rightarrow\ r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)$$
- $Z(x)$ cancels in the Bradley-Terry difference, so the reward model is reparameterized by the policy. Result (matches TRL docs):
  $$\mathcal L_{DPO}=-\mathbb E_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$$
- Implicit reward $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$. Gradient: $-\beta\,\mathbb E[\sigma(\hat r_l-\hat r_w)(\nabla\log\pi_\theta(y_w)-\nabla\log\pi_\theta(y_l))]$ - pairs the implicit reward currently ranks wrong get larger weight (paper's gradient form; stated from the standard derivation).
- Needs: offline pairs (prompt, chosen, rejected), a frozen reference (usually the SFT model; computed on the fly or precomputed). No sampling or RM during training. Paper limits: tested up to 6B; OOD generalization open.
- $\beta$: strength of the KL anchor. Large $\beta$ = stay close to reference (loss saturates after a small log-ratio change); small $\beta$ = drift more freely. TRL default `beta=0.1`, default `learning_rate=1e-6`, `loss_type="sigmoid"`; for adapters TRL docs suggest ~1e-5 LR. DPO paper's own beta/LR values: UNVERIFIED (commonly cited 0.1, lr 1e-6 style).
- TRL logs: `rewards/chosen`, `rewards/rejected` ($\beta\log$-ratios), `rewards/margins`, `rewards/accuracies`, `logps/chosen`. TRL notes DPO in practice works mostly by suppressing the rejected completion rather than raising the chosen one.

## Variants (formulas; all verified from the source papers unless marked)

| Method                                   | Loss / reward                                                                                                                                                                                                                          | Data                    | Ref model                           |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------- |
| DPO                                      | above                                                                                                                                                                                                                                  | pairs                   | yes                                 |
| IPO (Azar et al. 2023, 2310.12036)       | $\big(h_\pi-\tfrac1{2\tau}\big)^2$, $h_\pi=\log\frac{\pi(y_w)}{\pi(y_l)}-\log\frac{\pi_{ref}(y_w)}{\pi_{ref}(y_l)}$                                                                                                                    | pairs                   | yes                                 |
| KTO (Ethayarajh et al. 2024, 2402.01306) | $\mathbb E[\lambda_y - v(x,y)]$; $v=\lambda_D\sigma(\beta(r_\theta-z_0))$ if desirable, $\lambda_U\sigma(\beta(z_0-r_\theta))$ if undesirable; $r_\theta=\log\frac{\pi_\theta}{\pi_{ref}}$, $z_0\approx$ KL (batch-estimated, no grad) | unpaired thumbs-up/down | yes                                 |
| ORPO (Hong et al. 2024, 2403.07691)      | $\mathcal L_{SFT}+\lambda\mathcal L_{OR}$, $\mathcal L_{OR}=-\log\sigma\big(\log\frac{odds_\theta(y_w)}{odds_\theta(y_l)}\big)$, $odds=\frac{P}{1-P}$                                                                                  | pairs                   | no (single stage, replaces SFT+DPO) |
| SimPO (Meng et al. 2024, 2405.14734)     | $-\log\sigma\big(\tfrac{\beta}{                                                                                                                                                                                                        | y_w                     | }\log\pi_\theta(y_w)-\tfrac{\beta}{ | y_l | }\log\pi_\theta(y_l)-\gamma\big)$ | pairs | no  |

- IPO rationale: with near-deterministic preferences the BT reward gap grows without bound, driving $\pi(y_l)\to0$ and weakening the KL term ("the more deterministic the preferences, the weaker the KL regularisation"); regressing to a finite target $1/(2\tau)$ keeps it anchored. In TRL, `beta` plays the role of $\tau$ for `loss_type="ipo"`.
- KTO guidance: $\beta$ 0.01-0.10 for larger SFT'd models, 0.10-1.0 for small/un-SFT'd; default 0.1; keep $\lambda_D n_D/(\lambda_U n_U)\in[1,1.5]$; LR start 5e-6 (2-10x DPO's 5e-7); effective batch 8-128 (they used 32), microbatch >=2.
- ORPO: lambda 0.1 (Mistral-ORPO-alpha), 0.2 (Llama-2 7B), 0.25 (Phi-2); LR 8e-6, 10 epochs (stated for OPT experiments); reports 12.20% AlpacaEval 2.0 / 66.19% IFEval / 7.32 MT-Bench for Mistral-ORPO-alpha-class models on UltraFeedback.
- SimPO: reward = average log-prob (matches generation metric, removes ref model; memory/compute saving). Paper hyperparameters: beta 2.0-2.5, gamma 0.3-1.6 (recommended 0.5-1.5), LR 3e-7 to 1e-6 (Mistral-Base 3e-7, Mistral-Instruct 5e-7, Llama-3-Base 6e-7, Llama-3-Instruct 1e-6), batch 128, 1 epoch. Claims up to +6.4 AlpacaEval 2 / +7.5 Arena-Hard over DPO; Gemma-2-9B-it model 72.4% LC AlpacaEval 2.
- Other TRL loss types: hinge (RSO/SLiC), robust (label flip), bco_pair, apo_zero/down, sigmoid_norm (=length-normalized, SimPO-style), sft term mixing via `loss_type=[...]`/`loss_weights`, `ld_alpha` (LD-DPO).

## Length-normalized DPO (Tulu 3; Lambert et al. 2024, arXiv 2411.15124)

- Tulu 3 uses "the length-normalized variant of DPO proposed in Meng et al. 2024" (SimPO): replace log-ratios by per-token averages, i.e. $\beta\big(\tfrac{1}{|y_w|}\log\tfrac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\tfrac1{|y_l|}\log\tfrac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\big)$ (form inferred from the described variant; exact equation not retrieved, UNVERIFIED). Motivation: counter length bias of human and model preferences.
- Hyperparameters (Table 20): $\beta=5$ (large because per-token averages are small), LR 5e-7 (8B) / 2e-7 (70B), effective batch 128, 1 epoch, linear schedule, warmup 0.1, max length 2048.
- Data: 354,192 pairs total; mix sizes 271,409 (8B) / 334,302 (70B); incl. 41,635 cleaned UltraFeedback; on-policy pairs from Tulu-SFT vs other models. Followed by RLVR (PPO).

## Failure modes

- Chosen log-prob falls: DPO only constrains the margin, so both $\log\pi(y_w)$ and $\log\pi(y_l)$ often drop (mass moves elsewhere). Watch `logps/chosen` while `rewards/margins` rises.
- Likelihood displacement (Razin et al. 2024, arXiv 2410.08847, ICLR 2025): chosen likelihood decreases and mass can shift to opposite-meaning outputs ("No" vs "Never" raised "Yes"). Llama-3-8B-Instruct refusal rate 74.4% -> 33.4% in a safety DPO run. Cause: similar hidden embeddings of chosen/rejected; CHES score identifies culprit samples; filtering helps.
- Over-optimization / verbosity: length exploitation (motivates SimPO/Tulu normalization); off-policy data from other models; margins blow up when preferences deterministic (IPO argument).
- Mitigations seen in TRL/lit: SFT term on chosen (`rpo_alpha`-style/ `loss_type` mixing), small LR (1e-6..5e-7), 1 epoch, higher beta, data filtering, on-policy pairs.

## 2025-2026 practice

- Delta Learning Hypothesis (Geng et al., COLM 2025, arXiv 2507.06187): pair a weak-teacher output against a weaker one; the quality delta drives gains even when SFT on the same data hurts. 8B models trained on pairs from 3B vs 1.5B models matched Tulu 3 on an 11-benchmark suite.
- Olmo 3 (Team Olmo, arXiv 2512.13961, v2 Apr 2026): Think pipeline is SFT -> DPO -> RLVR, DPO data "following the insights from Delta Learning"; Instruct DPO uses delta-learning heuristic pairs and delta-aware GPT-judged pairs, multi-turn preferences, length-bias controls; then RL. Specific DPO beta/LR/data sizes and which model pairs: UNVERIFIED (section not retrieved).
- General trend (verified at the level of Tulu 3 and Olmo 3): DPO is the standard middle stage between SFT and RL; RL with verifiable rewards follows.

## Suggested visuals

1. Margin explorer: sliders for $\log\frac{\pi_\theta}{\pi_{ref}}$ of chosen and rejected and beta in {0.01,0.1,0.5}; show implicit rewards, margin, $\sigma$, loss, gradient weight $\sigma(-\Delta)$.
2. Training trajectories (illustrative, shape consistent with TRL-logged runs at beta 0.1, LR ~5e-7..1e-6): loss 0.693 -> ~0.45-0.55 over ~1 epoch; rewards/margins 0 -> 0.5-2; accuracies 50% -> 65-75%; rewards/chosen drifting slightly negative while rejected falls more; logps/chosen decreasing (the failure-mode panel).
3. Probability-mass picture: bar chart over 6 candidate responses; DPO lowers rejected and chosen, mass leaks to a third response (likelihood displacement), with a "similar embeddings" toggle.
4. Loss-comparison panel: DPO sigmoid vs IPO quadratic vs SimPO (with margin gamma) as functions of margin; show IPO's finite target and DPO's unbounded push.
5. Data-format table: pairs (DPO/IPO/SimPO/ORPO) vs thumbs (KTO) and ref-model toggle with memory cost (policy+ref = 2 x 16 GB bf16 for 8B).
6. Length-bias demo: chosen responses 2x longer; raw DPO vs normalized (beta 5); show effective per-token reward.

## References

- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. https://arxiv.org/abs/2305.18290
- Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M., Munos, R. (2023). A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO). https://arxiv.org/abs/2310.12036
- Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., Kiela, D. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. https://arxiv.org/abs/2402.01306
- Hong, J., Lee, N., Thorne, J. (2024). ORPO: Monolithic Preference Optimization without Reference Model. https://arxiv.org/abs/2403.07691
- Meng, Y., Xia, M., Chen, D. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward. https://arxiv.org/abs/2405.14734
- Lambert, N., et al. (2024). Tulu 3: Pushing Frontiers in Open Language Model Post-Training. https://arxiv.org/abs/2411.15124
- Razin, N., Malladi, S., Bhaskar, A., Chen, D., Arora, S., Hanin, B. (2024). Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization. https://arxiv.org/abs/2410.08847
- Geng, S., Ivison, H., Li, C.-L., Sap, M., Li, J., Krishna, R., Koh, P. W. (2025). The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains. https://arxiv.org/abs/2507.06187
- Team Olmo (2025). Olmo 3. https://arxiv.org/abs/2512.13961
- Hugging Face TRL, DPO Trainer docs. https://huggingface.co/docs/trl/main/en/dpo_trainer
