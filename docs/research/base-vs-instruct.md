# Base vs instruct models: research notes

Verified 2026-11-09 against arXiv PDFs (text-extracted), Hugging Face API/raw files and official docs. "UNVERIFIED" marks anything not checked against a primary source.

## 1. What a base model is

- A base model is a causal LM trained only to predict the next token over a huge text corpus. Loss over a sequence: $\mathcal{L}_{PT}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})$.
- HF docs: "All causal LMs, whether chat-trained or not, continue a sequence of tokens"; a chat model is a base model fine-tuned on data formatted as messages, and the chat is "still just a sequence of tokens" [HF-CT].
- Consequence: a prompt is treated as the start of a _document_, not as a request. The most likely continuation of a question in web text is often more questions, a forum reply, a quiz list, or a repeat of the question.
- InstructGPT motivation: the LM objective ("predict the next token on a webpage") differs from "follow the user's instructions helpfully and safely" [InstructGPT]. (Exact text of its GPT-3 failure examples not re-extracted: UNVERIFIED.)
- Realistic behaviours (ILLUSTRATIVE, written for the site; NOT recorded runs, so reproduce on a real checkpoint before publishing as "output"):
  - Prompt `What is the capital of France?` -> `What is the capital of Germany? What is the capital of Italy? ...` (list continuation) or `Paris is the capital...` (Wikipedia-style) depending on sampling.
  - Prompt `Write a poem about the sea.` -> `Write a poem about the moon. Write a poem about ...` (worksheet style).
  - Never stops: no end-of-turn habit, so it runs to `max_new_tokens` or emits `<|endoftext|>` at a document boundary.
- Few-shot / prefix trick: a base model answers well if the document _looks like_ a Q/A page (`Q: ...\nA:`). This is the bridge to SFT: SFT bakes that format in. In-context alignment evidence: URIAL [URIAL].

## 2. Pretraining -> mid-training (current practice)

Mid-training = a shorter second stage (typically 5-10% of FLOPs) on upsampled high-quality/math/code/reasoning/instruction-like data with LR annealed to ~0, plus context extension.

- **Llama 3** [Llama3]: final 40M tokens of pretraining linearly anneal LR to 0 at long context; annealing on small amounts of high-quality code/math data boosts benchmarks; annealing also used to judge data quality (Sec. 3.1.3).
- **OLMo 2** [OLMo2]: "mid-training" = second stage, 5-10% of training FLOPs, on Dolmino Mix 1124 (high-quality web, math, instruction-like data), citing Llama 3 and others as precedent.
- **Qwen3** [Qwen3]: 36T tokens, 119 languages. S1 general: >30T tokens at 4,096 seq len. S2 reasoning: more STEM/code/reasoning/synthetic data, faster LR decay. S3 long context: hundreds of billions of tokens at 32,768 (75% of docs 16K-32K, 25% 4K-16K); RoPE base 10,000 -> 1,000,000 (ABF); YaRN for inference-time extension.
- **SmolLM3** [SmolLM3]: 11.2T tokens in 3 stages (web/code/math: 85/12/3 -> 75/15/10 -> 63/24/13 %). Mid-training: long context 100B tokens in two 50B stages (4k->32k, RoPE theta 1.5M; 32k->64k, theta 5M; YaRN to 128k at inference); then reasoning mid-training on a 35B-token set (OpenThoughts3-1.2M + Llama-Nemotron subset) for 4 epochs (~140B tokens), ChatML template and wrapped packing.
- **Olmo 3** [Olmo3]: Stage 1 pretraining on Dolma 3 Mix (6T-token mix, 5.93T actual); Stage 2 midtraining 100B tokens on Dolma 3 Dolmino Mix; Stage 3 long-context extension on Dolma 3 Longmino Mix, 50B (7B) / 100B (32B) tokens. Midtraining includes post-training-style data (instruction and thinking traces) validated by 100B "integration tests".
- Olmo 3 trap worth showing: including chat special tokens (`<|im_start|>`, `<|im_end|>`) in midtraining data made the model emit them at inference and collapsed scores (GSM8K 49.43 -> 0, CruxEval 32.89 -> 18.91), while a plain-text template did not (46.02 / 29.65). So they leave special tokens for SFT [Olmo3, Sec. 3.5 "Leave special tokens for SFT stage"].
- Takeaway for the site: "base" checkpoints released in 2025 are already post-mid-training (annealed, often seen instruction/reasoning-like text), so they are less "raw web" than GPT-3 era bases and may sometimes answer questions.

## 3. Sampling (decoding) from logits $z_i$ over vocabulary $V$

- Temperature: $p_i = \dfrac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$. $T\to0$ greedy; $T=1$ model distribution; $T>1$ flatter.
- Top-k: keep the $k$ highest-probability tokens, zero the rest, renormalise.
- Top-p / nucleus [Nucleus]: smallest set $V_p$ with $\sum_{i\in V_p} p_i \ge p$ (tokens sorted by prob); renormalise within $V_p$.
- Min-p [MinP] (ICLR 2025 oral): keep tokens with $p_i \ge p_{\text{base}}\cdot p_{\max}$, where $p_{\max}=\max_j p_j$; threshold scales with model confidence. (Formula from the paper's method; the abs page only states the idea, so formula wording is UNVERIFIED against the PDF text.)
- Order in HF `generate` is configurable; common practice: temperature, then truncation (top-k/top-p/min-p), then sample.
- Official recommended settings (Qwen3-8B card): thinking mode T=0.6, top-p 0.95, top-k 20, min-p 0, "DO NOT use greedy decoding"; non-thinking T=0.7, top-p 0.8, top-k 20, min-p 0 [Qwen3-8B-card].

## 4. Concrete base/instruct pairs on Hugging Face (all HTTP 200 on the HF API, 2026-11-09)

| Family            | Base repo                       | Post-trained repo(s)                                                                                   |
| ----------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Qwen3             | `Qwen/Qwen3-8B-Base`            | `Qwen/Qwen3-8B` (hybrid think/no-think; no separate "Instruct" for dense 8B)                           |
| Olmo 3            | `allenai/Olmo-3-1025-7B`        | `allenai/Olmo-3-7B-Instruct`, `allenai/Olmo-3-7B-Think`; intermediate `allenai/Olmo-3-7B-Instruct-SFT` |
| Olmo 2            | `allenai/OLMo-2-1124-7B`        | `allenai/OLMo-2-1124-7B-Instruct` (the lowercase `Olmo-2-...` path 307-redirects)                      |
| SmolLM3           | `HuggingFaceTB/SmolLM3-3B-Base` | `HuggingFaceTB/SmolLM3-3B`                                                                             |
| Llama 3.1 (gated) | `meta-llama/Llama-3.1-8B`       | `meta-llama/Llama-3.1-8B-Instruct`                                                                     |

- Qwen3-8B: 8.2B params (6.95B non-embedding), 36 layers, GQA 32 Q / 8 KV heads, 32,768 native context (131,072 with YaRN) [Qwen3-8B-card]. EOS in tokenizer_config: base `<|endoftext|>`, post-trained `<|im_end|>` (checked). Base also ships a ChatML template in tokenizer_config but is not trained to follow it (the template alone does not make it a chat model; behaviour of the base on it not tested: UNVERIFIED).
- Olmo 3 naming: only 7B base is `-1025-`; 32B base is `Olmo-3-1125-32B` (from paper's link targets).

## Suggested visuals

1. "Same prompt, two models": side-by-side token streaming for `What is the capital of France?` from base vs instruct. Use real captured outputs from the pairs above (T=0.7, 64 new tokens); show base ending on `<|endoftext|>` or hitting max tokens, instruct ending on `<|im_end|>`.
2. Training-stage timeline (to scale): Qwen3 30T -> reasoning stage -> hundreds of B long ctx; Olmo 3 5.93T -> 100B -> 50B(7B); SmolLM3 11.2T -> 100B + 140B. Mid-training is ~1-3% of tokens but where the format starts to appear.
3. Sampler lab: one next-token distribution (e.g. top-10 probs 0.42, 0.21, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01). Sliders T, k, p, min-p; show the kept set and renormalised bars. With min-p 0.1: threshold $0.1\times0.42=0.042$, keeps 5 tokens. With p=0.9: keeps 0.42+0.21+0.12+0.08+0.05+0.04=0.92 -> 6 tokens.
4. Special-token trap: toggle `<|im_end|>` seen/unseen in midtraining and show the Olmo 3 GSM8K 49.43 -> 0 drop.

## References

- [HF-CT] Hugging Face, "Chat templates", Transformers docs. https://huggingface.co/docs/transformers/main/en/chat_templating
- [InstructGPT] Ouyang et al., "Training language models to follow instructions with human feedback", 2022. https://arxiv.org/abs/2203.02155
- [Llama3] Grattafiori et al., "The Llama 3 Herd of Models", 2024. https://arxiv.org/abs/2407.21783
- [OLMo2] Team OLMo, Walsh, Soldaini, Groeneveld et al., "2 OLMo 2 Furious", 2024. https://arxiv.org/abs/2501.00656
- [Qwen3] An Yang, Anfeng Li, Baosong Yang et al., "Qwen3 Technical Report", 2025. https://arxiv.org/abs/2505.09388
- [Olmo3] Team Olmo (Ettinger, Bertsch, Kuehl, Graham, Heineman et al.), "Olmo 3", Dec 2025 (v2 Apr 2026). https://arxiv.org/abs/2512.13961
- [SmolLM3] Bakouch et al., "SmolLM3: smol, multilingual, long-context reasoner", Hugging Face blog, 2025. https://huggingface.co/blog/smollm3 (authors list not extracted; Bakouch per Olmo 3 citation)
- [Nucleus] Holtzman, Buys, Du, Forbes, Choi, "The Curious Case of Neural Text Degeneration", 2019. https://arxiv.org/abs/1904.09751
- [MinP] Nguyen, Baker, Neo, Roush et al., "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs", 2024 (ICLR 2025). https://arxiv.org/abs/2407.01082
- [URIAL] Lin et al., "The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning", 2023. https://arxiv.org/abs/2312.01552 (author list not checked)
- [Qwen3-8B-card] https://huggingface.co/Qwen/Qwen3-8B ; base: https://huggingface.co/Qwen/Qwen3-8B-Base
- Olmo 3: https://huggingface.co/allenai/Olmo-3-1025-7B , https://huggingface.co/allenai/Olmo-3-7B-Instruct ; SmolLM3: https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base
