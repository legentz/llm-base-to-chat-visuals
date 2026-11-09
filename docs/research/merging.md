# Model merging: weight-space arithmetic

Verified 2026-11-09 against arXiv full text. Toy numbers below are my own illustrations (marked "toy"), not from the papers.

## 0. Setup

Pretrained weights $\theta_0$; fine-tuned models $\theta_t$ (same architecture, same init). Task vector $\tau_t=\theta_t-\theta_0$. All methods below operate parameter-wise on these, with no gradient steps and no GPU needed in the simplest cases (DARE paper title: "free lunch").

## 1. Model soups (Wortsman et al. 2022, arXiv 2203.05482)

- Uniform soup: $\theta_S=\frac1{|S|}\sum_{i\in S}\theta_i$, averaging fine-tuned models from a hyper-parameter sweep (learning rate, weight decay, augmentation, ...). Accuracy goes up with no extra inference cost.
- Greedy soup: sort by held-out accuracy, add a model only if validation accuracy does not decrease.
- Result: ViT-G soup reaches 90.94% top-1 on ImageNet (new state of the art at the time).
- Condition: models start from the same pretrained init and sit in one loss basin.

## 2. Task arithmetic (Ilharco et al. 2023, arXiv 2212.04089, ICLR 2023)

- $\tau_t=\theta_t-\theta_0$; edit by arithmetic: addition $\theta=\theta_0+\lambda\sum_t\tau_t$ (multi-task), negation $\theta=\theta_0-\lambda\tau$ (forget/unlearn a behaviour, e.g. toxicity), with scaling $\lambda$ picked on held-out data.
- Paper: negating a task vector lowers target-task performance with little change on control tasks; adding vectors can improve several tasks at once. They also tried 427 candidate checkpoints found on Hugging Face Hub.
- Toy: $\theta_0=[1.0,1.0,1.0]$, $\tau_A=[+0.2,0,-0.1]$, $\tau_B=[0,+0.3,+0.1]$, $\lambda=1$ gives $\theta=[1.2,1.3,1.0]$. Here the $-0.1$ and $+0.1$ cancel on the third coordinate (interference).
- Real HF example of the same idea: "chat vector". Card of `hon9kon9ize/CantoneseLLM-v2.0-8B-Thinking-Chat-Vector-Merged` (`base_model_relation: merge`) says it takes Qwen3-8B-Base, continues pre-training it on 784M tokens of Cantonese/HK text, then adds the chat vector taken from Qwen3-8B, i.e. $\theta=\theta_{cpt}+(\theta_{Qwen3\text{-}8B}-\theta_{Qwen3\text{-}8B\text{-}Base})$.

## 3. SLERP (Shoemake 1985; used in mergekit)

Interpolate along the great-circle arc rather than the chord. For weight vectors $w_0,w_1$ (flattened per tensor) with angle $\Omega=\arccos\frac{w_0\cdot w_1}{\|w_0\|\|w_1\|}$:
$$\mathrm{slerp}(w_0,w_1;t)=\frac{\sin((1-t)\Omega)}{\sin\Omega}w_0+\frac{\sin(t\Omega)}{\sin\Omega}w_1$$

- Mergekit paper: SLERP is "an extension of simple weight averaging", with the claim that there is often a spherical path with a lower loss barrier than direct linear interpolation. Works on two models only.
- Toy: $w_0=(1,0)$, $w_1=(0,1)$, $t=0.5$. LERP gives $(0.5,0.5)$ with norm 0.707. SLERP gives $(0.707,0.707)$ with norm 1.0, so the norm is preserved.
- Mergekit allows $t$ to vary by layer and by module (e.g. different $t$ for attention and MLP): a gradient across depth. The exact YAML syntax is not given here.

## 4. TIES-merging (Yadav et al. 2023, arXiv 2306.01708)

Problem: redundant small updates and sign conflicts between task vectors interfere. Three steps:

1. Trim: keep the top-$k\%$ magnitude entries of each $\tau_t$, zero the rest, giving $\hat\tau_t=\hat\gamma_t\odot\hat\mu_t$ (signs times magnitudes).
2. Elect sign: $\gamma_m^p=\mathrm{sgn}\big(\sum_t\hat\tau_t^p\big)$, i.e. the sign with the larger total mass.
3. Disjoint merge: $\tau_m^p=\frac1{|A^p|}\sum_{t\in A^p}\hat\tau_t^p$, $A^p=\{t:\mathrm{sgn}(\hat\tau_t^p)=\gamma_m^p\}$. Mean over agreeing, nonzero entries only.
   Then $\theta_m=\theta_0+\lambda\tau_m$.

- Toy (2 tasks, 4 params, keep top 50%): $\tau_1=[0.8,-0.1,0.5,0.05]$, $\tau_2=[0.6,0.3,-0.4,-0.02]$. After trim: $[0.8,0,0.5,0]$ and $[0.6,0,-0.4,0]$. Elected signs: $[+,\cdot,+,\cdot]$ (coordinate 3 sums to +0.1). Disjoint mean: $[0.7,0,0.5,0]$. Plain average: $[0.7,0.1,0.05,0.015]$, which has nearly cancelled coordinate 3 and keeps noise on coordinates 2 and 4.

## 5. DARE (Yu et al. 2023, "Language Models are Super Mario", arXiv 2311.03099)

- Drop And REscale on delta parameters: sample mask $m\sim\mathrm{Bernoulli}(1-p)$ and
  $$\tilde\tau=\frac{m\odot\tau}{1-p}$$
  so $\mathbb E[\tilde\tau]=\tau$. The paper reports eliminating up to 90% or even 99% of the deltas of SFT models (e.g. WizardMath on GSM8K) with little loss; rescaling is "crucial", since dropping 30-40% without rescaling hurts noticeably.
- Used as a pre-step to reduce interference before a sum or TIES. Mergekit exposes `dare_ties` and `dare_linear` (the paper lists DARE-TIES among its methods; `dare_linear` name is from memory, UNVERIFIED).
- Toy: $p=0.5$, $\tau=[0.4,0.2,0.6,0.8]$, mask keeps coordinates 1 and 3 gives $\tilde\tau=[0.8,0,1.2,0]$ (expected value preserved over random masks).
- Premise: SFT deltas are small and redundant (applies to SFT deltas, not to large continued-pretraining deltas: UNVERIFIED nuance).

## 6. mergekit (Goddard et al. 2024, arXiv 2403.13257, Arcee)

- Open-source toolkit for merging on CPU or low-memory GPU. Merging happens in parameter space; supports Linear (LERP/average), SLERP, task arithmetic, TIES, DARE-TIES, and "passthrough" (layer stacking, nicknamed FrankenMerging, producing models of a new size).
- Paper example: merging Meditron-7B with Llama-2-7B chat via LERP, SLERP, TIES and DARE-TIES.
- HF integration: merged repos usually carry the `mergekit` and `merge` tags and `base_model_relation: merge`, with all parents in `base_model`. See case-study.md for a live example.
- Evolutionary merging (EvoLLM-JP, searching parameter-space and data-flow-space recipes) is cited in the mergekit paper.

## 7. Comparison cheat sheet

| Method              | Inputs                    | Key idea                        | Hyper-params    |
| ------------------- | ------------------------- | ------------------------------- | --------------- |
| Uniform/greedy soup | same-init fine-tunes      | average / validate-then-add     | none / val set  |
| Task arithmetic     | $\theta_0$ + task vectors | add or negate $\lambda\tau$     | $\lambda$       |
| SLERP               | 2 models                  | spherical interpolation         | $t$ (per layer) |
| TIES                | $\theta_0$ + n vectors    | trim, elect sign, disjoint mean | $k$, $\lambda$  |
| DARE                | $\theta_0$ + vectors      | random drop, rescale $1/(1-p)$  | $p$             |

## Suggested visuals

1. Vector diagram: $\theta_0$ at the origin, two arrows $\tau_A,\tau_B$ and their sum, then the negation arrow.
2. Chord vs arc: two points on a circle with LERP midpoint inside the circle (norm 0.707) and SLERP on it (norm 1.0). Slider for $t$.
3. TIES animation on a 4x2 grid with the numbers from the toy: original, trimmed, elected-sign column, disjoint mean versus average.
4. DARE: 20 weights shown as bars; drop 90% (keep 2 bars), rescale them by 10, and show the sum is preserved in expectation.
5. Model tree strip tying to case-study.md: Qwen3-8B-Base -> CPT -> add chat vector -> merged model.
6. Soup lineup: 5 fine-tune accuracies, e.g. 80.1, 80.4, 79.8, 80.6, 79.5 with the soup at about 81.3 (illustrative only).

## References

1. Wortsman, Ilharco, Gadre, Roelofs, Gontijo-Lopes, Morcos, Namkoong, Farhadi, Carmon, Kornblith, Schmidt. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." 2022. https://arxiv.org/abs/2203.05482
2. Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi. "Editing Models with Task Arithmetic." ICLR 2023. https://arxiv.org/abs/2212.04089
3. Yadav, Tam, Choshen, Raffel, Bansal. "TIES-Merging: Resolving Interference When Merging Models." 2023. https://arxiv.org/abs/2306.01708
4. Yu, Yu, Yu, Huang, Li. "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch." 2023. https://arxiv.org/abs/2311.03099
5. Goddard, Siriwardhana, Ehghaghi, Meyers, Karpukhin, Benedict, McQuade, Solawetz. "Arcee's MergeKit: A Toolkit for Merging Large Language Models." 2024. https://arxiv.org/abs/2403.13257
6. Shoemake. "Animating rotation with quaternion curves." SIGGRAPH 1985 (SLERP; cited by the mergekit paper, not re-read).
7. Model card: https://huggingface.co/hon9kon9ize/CantoneseLLM-v2.0-8B-Thinking-Chat-Vector-Merged
