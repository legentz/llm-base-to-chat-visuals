# LoRA, QLoRA, DoRA: research notes

Verified 2026-11-09 against arXiv abs/html pages and HF PEFT docs. "UNVERIFIED" = not confirmed from a primary source this session. "(derived)" = my arithmetic, not a paper claim.

## LoRA (Hu et al., 2021, arXiv 2106.09685)

- Freeze pretrained W; learn low-rank update. Forward: $h = W x + \frac{\alpha}{r} B A x$, so $W' = W + \frac{\alpha}{r} BA$, with $B\in\mathbb{R}^{d_{out}\times r}$, $A\in\mathbb{R}^{r\times d_{in}}$, $r \ll \min(d_{in},d_{out})$.
- Init: paper: "random Gaussian initialization for A and zero for B", so $\Delta W = BA = 0$ at step 0 (the model starts exactly equal to the base).
- PEFT default (`init_lora_weights=True`): A Kaiming-uniform, B zero. Option `"gaussian"` = Gaussian A, zero B.
- Scaling: $\alpha$ is a constant in $r$; the authors set it to the first r they tried and did not tune it, to reduce retuning when r changes.
- Paper's experiments adapt only $W_q$ and $W_v$ (MLP frozen). GPT-3 175B, r=4 on Wq/Wv: checkpoint 350 GB -> 35 MB (~10,000x); training VRAM ~1.2 TB -> 350 GB (~3x). No inference latency: merge $W+\frac{\alpha}{r}BA$.
- Modern practice: adapt ALL linear layers (QLoRA paper: Wq/Wv-only did not match full FT; "LoRA on all linear transformer block layers [is] required"; Thinking Machines agrees, see below). PEFT: `target_modules="all-linear"` (excludes output head).
- PEFT defaults: `r=8`, `lora_alpha=8`, `lora_dropout=0.0`, `use_rslora=False`, `use_dora=False`.
- Param count per adapted matrix: $r\,(d_{in}+d_{out})$ vs $d_{in}d_{out}$ for full.
- Worked count (derived; assumes Llama-3-8B dims: d=4096, 32 layers, KV dim 1024, FFN 14336, dims UNVERIFIED this session): per layer q 8192r + k 5120r + v 5120r + o 8192r + gate/up/down 3x18432r = 81,920 r; x32 layers = 2,621,440 r. r=16 -> 41.9M (0.52% of ~8.03B); r=64 -> 168M; r=1 -> 2.6M.

## rsLoRA (Kalajdzievski, 2023, arXiv 2312.03732)

- Claim: standard $\alpha/r$ slows learning / hurts at high rank; adapters should be divided by $\sqrt r$: $\gamma_r=\alpha/\sqrt r$. Lets large ranks help at no inference cost.
- PEFT: `use_rslora=True` -> `scaling = lora_alpha / sqrt(r)`; default `lora_alpha / r`.
- Caveat: Thinking Machines (below) reports that with alpha=32 and the 1/r convention the optimal LR is roughly rank-independent, so the two conventions are not obviously in conflict; the useful bit is "which scaling you use changes how LR transfers across ranks".

## QLoRA (Dettmers, Pagnoni, Holtzman, Zettlemoyer, 2023, arXiv 2305.14314)

- Backprop through a frozen 4-bit quantized base into LoRA adapters. 65B finetune: >780 GB -> <48 GB (single 48 GB GPU).
- NF4 (4-bit NormalFloat): 16 levels = quantiles of N(0,1) rescaled to [-1,1]; asymmetric so zero is exact; information-theoretically good for normally distributed weights.
- Double quantization: quantize the per-block quantization constants; overhead 0.5 -> ~0.127 bits/param (saves ~0.373 bits, ~3 GB at 65B). So NF4+DQ ~ 4.127 bits/param.
- Paged optimizers: NVIDIA unified memory pages optimizer state CPU<->GPU to survive memory spikes (e.g., gradient checkpointing with long sequences).
- Compute: weights dequantized on the fly to bf16 for matmul; adapters and grads in bf16/fp32. (4-bit compute dtype detail not verified from text; UNVERIFIED.)
- Their chatbot recipe (App. B.2): r=64, alpha=16, all linear layers; LR 2e-4 (7B/13B), 1e-4 (33B/65B); dropout 0.1 (<=13B), 0.05 (33B/65B); Adam beta2 0.999, max grad norm 0.3.
- Guanaco 65B reached 99.3% of ChatGPT on Vicuna benchmark after 24 h on one GPU (abstract claim).

## DoRA (Liu et al., 2024, arXiv 2402.09353; ICML 2024 oral)

- Decompose $W = m \,\frac{V}{\lVert V\rVert_c}$ (magnitude vector $m$, direction $V$, column-wise norm). Train $m$ directly; apply LoRA only to the direction: $W' = m\,\frac{W_0 + BA}{\lVert W_0+BA\rVert_c}$.
- Claim: better capacity/stability than LoRA, no extra inference cost after merge (LLaMA, LLaVA, VL-BART). PEFT: `use_dora=True`; docs note more training overhead than LoRA.

## Memory arithmetic, 8B model (derived; N = 8.03e9 params, activations/KV/CUDA overhead excluded)

Bytes per parameter:

| Item                                                                                                                                                                                                                                                                                                                                     | Full FT (bf16 + fp32 Adam) | LoRA (bf16 base)  | QLoRA (NF4 base)   |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ----------------- | ------------------ |
| Base weights                                                                                                                                                                                                                                                                                                                             | 2                          | 2                 | ~0.516 (4.127 bit) |
| Grads (trainable only)                                                                                                                                                                                                                                                                                                                   | 2                          | ~0 (adapter only) | ~0                 |
| Adam m + v (fp32)                                                                                                                                                                                                                                                                                                                        | 8                          | ~0 (adapter only) | ~0                 |
| fp32 master copy (optional)                                                                                                                                                                                                                                                                                                              | 4                          | -                 | -                  |
| Totals for 8.03B: full FT 12 B/param = 96 GB (+32 GB if fp32 master = 128 GB, i.e. the "16 B/param" rule); LoRA r=16: 16.1 GB base + adapters 41.9M x (4 weight + 4 grad + 8 Adam = 16 B) = 0.67 GB -> ~16.8 GB; QLoRA: ~4.1 GB base (embeddings/lm_head usually kept in 16-bit, so ~5 GB in practice, UNVERIFIED) + 0.67 GB -> ~5-6 GB. |
| Take-away for visual: optimizer state, not weights, dominates full FT (8 of 12-16 B); LoRA deletes it, QLoRA additionally shrinks weights 4x. Activations then dominate; gradient checkpointing and sequence length matter.                                                                                                              |

## 2025-2026 findings: "LoRA Without Regret" (Schulman et al., Thinking Machines Lab, 2025-09-29; blog, not arXiv) - https://thinkingmachines.ai/blog/lora/

- SFT on small/medium instruction & reasoning sets (Tulu3, OpenThoughts3): LoRA ~ same sample efficiency and final performance as full FT. Underperforms only when dataset exceeds adapter capacity.
- Best LR for LoRA ~10x full-FT LR (fitted 9.8; ~15x for runs <100 steps, preliminary).
- Apply to all layers, esp. MLP/MoE; attention-only underperforms even at matched params (attn-only r=256, 0.25B < MLP-only r=128, 0.24B).
- RL: LoRA matches full FT even at rank 1 (authors estimate RL on MATH absorbs ~320k bits vs ~3M params for rank-1 on Llama-3.1-8B).
- Large batch sizes hurt LoRA more than full FT; higher rank does not fix it.
- alpha=32 with 1/r makes optimal LR approx rank-independent. LoRA uses a bit over 2/3 of FullFT FLOPs per pass.

## Suggested visuals

1. Matrix animation: W (4096x4096 grid, 16.8M cells) next to B (4096x r) and A (r x 4096); slider r in {1,4,16,64,256}; live counter of trainable params and % of model (4096x4096 at r=16: 131k = 0.78%).
2. Init demo: B=0 so BA=0, output identical at step 0; show how $\Delta W$ grows during training (heatmap of BA, effective rank via singular values).
3. Memory stacked bar for 8B: full FT (16/16/64 + master 32 GB) vs LoRA (16 + 0.7) vs QLoRA (4.1 + 0.7), toggle precision, optimizer (AdamW/8-bit/SGD), show 24 GB / 80 GB GPU lines.
4. NF4 vs uniform-int4 level histogram over a Gaussian weight sample; show quantization error; double-quant bits/param toggle (0.5 vs 0.127).
5. Alpha/r vs alpha/sqrt(r) scaling table across r with fixed alpha; "effective update magnitude" chart (illustrative only).
6. Target-modules picker (q,v / attention / all-linear) updating param count and a schematic quality indicator citing QLoRA/Thinking Machines (qualitative only).

## References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. https://arxiv.org/abs/2305.14314
- Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., Chen, M.-H. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. https://arxiv.org/abs/2402.09353
- Kalajdzievski, D. (2023). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. https://arxiv.org/abs/2312.03732
- Schulman, J. et al., Thinking Machines Lab (2025). LoRA Without Regret. https://thinkingmachines.ai/blog/lora/
- Hugging Face PEFT docs, LoRA reference. https://huggingface.co/docs/peft/main/en/package_reference/lora
