# Supervised fine-tuning (SFT): research notes

Verified 2026-11-09 against HF raw `tokenizer_config.json` / `chat_template.jinja`, TRL docs, and arXiv PDFs (text-extracted). Superscript exponents are sometimes lost in PDF extraction; those values are flagged "(exponent inferred)".

## 1. Chat templates and special tokens

A chat template is a Jinja string in the tokenizer; `tokenizer.apply_chat_template(messages, tokenize=..., add_generation_prompt=...)` turns `[{"role","content"}]` into one token sequence [HF-CT]. `add_generation_prompt=True` appends the assistant header so the model replies instead of continuing the user text; use `False` for training. With `tokenize=False` then re-tokenizing, pass `add_special_tokens=False` to avoid duplicated BOS/EOS [HF-CT].

- **ChatML (Qwen3, SmolLM3, Olmo 3)**, from `Qwen/Qwen3-8B` tokenizer_config (eos = `<|im_end|>`):

```
<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n
```

Generation prompt = `<|im_start|>assistant\n`. Qwen3 base (`Qwen3-8B-Base`) has the same template but eos `<|endoftext|>`; TRL warns to align `eos_token` with the template (`eos_token="<|im_end|>"`) when SFT-ing Qwen bases [TRL-SFT].

- **Llama 3.x**, from a mirror of the Llama-3.1-8B-Instruct template (`unsloth/Llama-3.1-8B-Instruct`; the gated `meta-llama` raw file was not fetchable): bos `<|begin_of_text|>`, per message `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`, generation prompt `<|start_header_id|>assistant<|end_header_id|>\n\n`, eos `<|eot_id|>`; tool/ipython turns can end with `<|eom_id|>`.
- **Qwen3 thinking**: default `enable_thinking=True` -> model emits `<think>...</think>` then answer. `enable_thinking=False` makes the template append `<think>\n\n</think>\n\n` after the assistant header (empty think block), a hard switch. Soft switches `/think` and `/no_think` in user or system turns work only when `enable_thinking=True`; last instruction wins in multi-turn [Qwen3-8B-card]. In the template, `<think>` content of previous assistant turns is dropped; only turns after the last user query keep reasoning (template logic with `last_query_index`). Qwen3 report: "thinking mode fusion" SFT stage mixes thinking and non-thinking data under these flags, plus a thinking-budget mechanism [Qwen3].
- **SmolLM3**: ChatML; system prompt gets `## Metadata ... Reasoning Mode: /think` or `/no_think` (default think; `enable_thinking`, or `/think`/`/no_think` inside the system message); `/system_override` disables the default system block (chat_template.jinja). Its template contains `{% generation %}` markers usable for assistant-only loss [TRL-SFT].
- **Olmo 3 Instruct**: ChatML-style `<|im_start|>`/`<|im_end|>` (chat_template.jinja), eos in tokenizer_config `<|endoftext|>`; default system text about being a function-calling assistant when none given. Olmo 3 paper: special chat tokens deliberately kept out of midtraining, introduced at SFT [Olmo3].

## 2. SFT objective

Concatenate prompt $c$ (system+user tokens, templates included) and response $y$ into $x=(c,y)$; maximise likelihood of response tokens only:
$$\mathcal{L}_{SFT}(\theta) = -\frac{1}{\sum_t m_t}\sum_{t=1}^{|x|} m_t \log p_\theta(x_t\mid x_{<t}),\qquad m_t=\begin{cases}1 & x_t\in \text{assistant turn (incl. }\texttt{<|im\_end|>})\\ 0 & \text{otherwise}\end{cases}$$

- Same cross-entropy as pretraining; only the data format and the mask change. Masked positions get label `-100` (ignore index) [TRL-SFT]. Learning to emit the end-of-turn token is part of the loss, which is what makes the model stop.
- TRL (`SFTConfig`, docs v1.17.1 page): `assistant_only_loss=True` (default False; conversational data; needs `{% generation %}`/`{% endgeneration %}` in the template, auto-patched for known families such as Qwen3); `completion_only_loss` (default None -> loss on completion for prompt-completion datasets, full sequence for plain text); `packing=False`, `packing_strategy="bfd"` (also `"bfd_split"`, `"wrapped"`), `padding_free=False`, `max_length=1024`, `learning_rate=2e-5`, `num_train_epochs=3.0`, `loss_type="nll"` (alt `"dft"`).
- Normalisation matters: Tulu 3 switched from mean to **sum** loss so every token weighs equally under gradient accumulation / multi-device averaging, and re-tuned LR (best 5e-6) [Tulu3].
- Packing: concatenate several examples into fixed-length sequences to remove padding. TRL `bfd` = best-fit decreasing (truncates overflow, implies padding-free); `wrapped` cuts mid-sequence. Cross-contamination risk is avoided with attention masking / padding-free position ids; Olmo 3 trains SFT with document packing (not padding) at 32,768 seq len [Olmo3]; SmolLM3 SFT uses BFD packing [SmolLM3].

## 3. Datasets and recipes (numbers as stated in sources)

| Source                                                                                                                                                                                                                                             | Data                                                                                                                                                  | Hyperparameters                                                                                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FLAN [FLAN]                                                                                                                                                                                                                                        | 137B LM, 62 public text datasets, 10 hand-written instruction templates per dataset; held-out task clusters                                           | see paper                                                                                                                                                                                                                 |
| InstructGPT [InstructGPT]                                                                                                                                                                                                                          | SFT set ~13k training prompts from labelers + API; 40 contractors                                                                                     | 16 epochs, cosine LR to 10%, dropout 0.2; LR 9.65e-6 (1.3B/6B, bs 32), 5.03e-6 (175B, bs 8). SFT is step 1 of 3 (then RM, PPO)                                                                                            |
| LIMA [LIMA]                                                                                                                                                                                                                                        | 1,000 curated prompt/response pairs (~750k tokens) on LLaMA 65B                                                                                       | 15 epochs, AdamW (0.9, 0.95), wd 0.1, LR 1e-5 linearly to 1e-6, bs 32, trim at 2048 tokens, residual dropout                                                                                                              |
| Tulu 3 [Tulu3]                                                                                                                                                                                                                                     | SFT mix from a 23.3M-prompt pool, ~939k prompts in the SFT column of Table 7 (column header inferred); personas, math, code, IF, safety, multilingual | 2 epochs, eff. batch 128, max len 4096, LR 5e-6 (8B) / 2e-6 (70B), linear schedule, sum loss; 8B on 32 GPUs ~6 h                                                                                                          |
| SmolTalk (SmolLM2) [SmolLM2]                                                                                                                                                                                                                       | HF instruction dataset `HuggingFaceTB/smoltalk`                                                                                                       | 2 epochs, global batch 128, seq 8192, LR 3e-4 (1.7B-scale; paper value)                                                                                                                                                   |
| SmolLM3 [SmolLM3]                                                                                                                                                                                                                                  | 1.8B SFT tokens (1B non-reasoning, 0.8B reasoning), 12+10 datasets                                                                                    | 4 epochs (~8B tokens), BFD packing, loss masked on user turns and tool results; LR not stated; then APO (Tulu3 prefs + synthetic Qwen3-32B vs 0.6B pairs); linear merge 0.9 APO soup + 0.1 mid-train ckpt                 |
| Olmo 3 [Olmo3]                                                                                                                                                                                                                                     | Dolci-Think-SFT, Dolci-Instruct-SFT (Instruct base mix ~100K examples from OLMo 2 mix + function calling, IF, WildChat; GPT-4.1 completions)          | 2 epochs, packing, 32K seq; batch 1M tokens (7B) / 4M (32B). 7B Think: 45.4B tokens, LR 5e-5; 32B Think: 45.2B tokens, LR 1e-4 souped with 5e-5; 7B Instruct: 3.4B tokens, LR 8e-5 (exponents inferred from table layout) |
| Rules of thumb for the site: full-parameter SFT LR is 1e-6 to 1e-4 (smaller for bigger models); 1-4 epochs for large mixes (Tulu 3 found longer did not help), up to 15-16 epochs for tiny sets (LIMA, InstructGPT 13k). LoRA: TRL suggests ~1e-4. |

## 4. Superficial Alignment Hypothesis and later evidence

- LIMA: "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used"; 1,000 examples were enough to compete with much more RLHF'd systems [LIMA].
- Supporting: URIAL shows base LLMs can be aligned with in-context examples and a system prompt, no tuning [URIAL] (not re-read in detail: UNVERIFIED claims beyond title).
- Qualifying: Tulu 3 finds performance keeps improving with more SFT data, large gains on GSM8K (math/reasoning/code are not "style"), though TruthfulQA drops [Tulu3]. Olmo 3 and SmolLM3 put 35B-100B+ tokens of instruction/reasoning data into mid-training, and Olmo 3 Think SFT uses 45B tokens (vs 750k for LIMA) [Olmo3, SmolLM3]. Reading: style is cheap, skills (reasoning, tools, long context) are not.

## 5. What changed in 2025-2026

- **Hybrid reasoning chat models**: one checkpoint with think/no-think switch (Qwen3 `enable_thinking`, SmolLM3 `/think`), vs separate Instruct and Think releases (Olmo 3) [Qwen3, SmolLM3, Olmo3].
- **SFT on reasoning traces** is large (tens of B tokens, distilled long CoT), then DPO (Olmo 3 uses "delta learning", chosen/rejected from strong vs weak models) and RLVR (verifiable rewards); small models also trained via distillation rather than the full 4-stage pipeline (Qwen3 notes 1/10 GPU hours for strong-to-weak distillation) [Olmo3, Qwen3].
- **Mid-training absorbs instruction-like data**; chat special tokens held back until SFT [Olmo3].
- **Tooling**: packing with padding-free/BFD, `assistant_only_loss` with `{% generation %}` templates, DFT loss option, `chat_template_path` to graft a template onto a base (TRL example turns `Qwen/Qwen3-0.6B-Base` into an instruct model with the SmolLM3 template on Capybara) [TRL-SFT]. Olmo 3 reports 8x SFT throughput moving from open-instruct to OLMo-core with token-based batches and packing.
- Model souping of SFT checkpoints (Olmo 3 Think, SmolLM3 merge) is now routine.

## Suggested visuals

1. **Template inspector**: type a chat; show raw string and token chips for ChatML vs Llama 3 side by side, special tokens coloured; toggle `add_generation_prompt`, `enable_thinking` (shows the injected `<think>\n\n</think>\n\n`). Realistic example: user "What is 2+2?" -> ~20 tokens.
2. **Loss-mask view**: same conversation, tokens greyed (masked, label -100) for system/user, lit for assistant incl. `<|im_end|>`; live mean-vs-sum loss readout. Toggle `assistant_only_loss` and show fraction of tokens contributing (e.g. 30/120).
3. **Packing animation**: 8 examples of length 120-900 into 2,048 slots: padding waste (~50-60% of slots) vs BFD packing (<5%).
4. **Training curve toy**: loss drops from ~2.0 to ~1.0 over 2 epochs; compare data size 1k (LIMA, 15 epochs) vs 1M (Tulu-like, 2 epochs), and LR 5e-6 vs 5e-5 divergence. Numbers illustrative.
5. **Before/after next-token bars**: after `<|im_start|>assistant\n`, the base puts mass on arbitrary text, the SFT model on `The`/`Paris`; and after the answer, mass on `<|im_end|>` rises to ~1.

## References

- [HF-CT] Hugging Face, "Chat templates". https://huggingface.co/docs/transformers/main/en/chat_templating
- [TRL-SFT] Hugging Face TRL, "SFT Trainer" (docs v1.17.1 page). https://huggingface.co/docs/trl/sft_trainer
- [Qwen3-8B-card] Qwen team, Qwen3-8B model card. https://huggingface.co/Qwen/Qwen3-8B (raw template: https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json)
- [Qwen3] An Yang, Anfeng Li, Baosong Yang et al., "Qwen3 Technical Report", 2025. https://arxiv.org/abs/2505.09388
- [InstructGPT] Ouyang, Wu, Jiang, Almeida et al., "Training language models to follow instructions with human feedback", 2022. https://arxiv.org/abs/2203.02155
- [FLAN] Wei, Bosma, Zhao, Guu et al., "Finetuned Language Models Are Zero-Shot Learners", 2021. https://arxiv.org/abs/2109.01652
- [LIMA] Zhou et al., "LIMA: Less Is More for Alignment", 2023. https://arxiv.org/abs/2305.11206 (title/authors from memory, content read from PDF)
- [Tulu3] Lambert, Morrison, Pyatkin, Huang et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training", 2024. https://arxiv.org/abs/2411.15124
- [SmolLM2] Ben Allal, Lozhkov, Bakouch, Blazquez, Penedo et al., "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model", 2025. https://arxiv.org/abs/2502.02737
- [SmolLM3] "SmolLM3: smol, multilingual, long-context reasoner", Hugging Face blog, 2025. https://huggingface.co/blog/smollm3
- [Olmo3] Team Olmo (Ettinger, Bertsch, Kuehl, Graham, Heineman et al.), "Olmo 3", 2025 (v2 Apr 2026). https://arxiv.org/abs/2512.13961
- [URIAL] Lin et al., "The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning", 2023. https://arxiv.org/abs/2312.01552
- [DFT] "On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification", 2025. https://arxiv.org/abs/2508.05629
- Llama 3 template mirror: https://huggingface.co/unsloth/Llama-3.1-8B-Instruct ; SmolLM3 template: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja
