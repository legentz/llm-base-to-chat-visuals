# Case study: a Qwen3-8B family tree on Hugging Face

All repo ids below were queried through the public HF API (`/api/models/<id>` and `?filter=base_model:<relation>:<id>`) on 2026-11-09 and exist. Download counts are snapshots and change daily.

## 1. How HF records lineage

- A model card's YAML `base_model:` (string or list) names parents. HF turns it into tags `base_model:<parent>` plus `base_model:<relation>:<parent>`.
- Relation is `finetune`, `adapter`, `merge` or `quantized`. It comes from the optional `base_model_relation` field, otherwise HF infers it (e.g. GGUF/AWQ/FP8 files are tagged `quantized`; PEFT repos `adapter`). The inference rules are not documented in what I read: UNVERIFIED.
- The "Model tree" widget on a repo page is built from those tags. It is only as good as the uploader's metadata. Gaps seen below:
  - `Qwen/Qwen3-4B-Instruct-2507` and `p-e-w/Qwen3-4B-Instruct-2507-heretic` carry no `base_model` tag (the Heretic card links its parent in prose only).
  - `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` has no `base_model` tags (YAML only has license and library). The card text says it was post-trained from Qwen3 8B Base using R1-0528 chain-of-thought, so the distillation edge is invisible in the tree.
  - The lineage API is queried per relation, so one parent can have thousands of children in each category.

## 2. The tree

```
Qwen/Qwen3-8B-Base                       (pre-trained; apache-2.0; created 2025-04-28)
 |-- finetune --> Qwen/Qwen3-8B          (post-trained chat/thinking model; 6.8M downloads)
 |                 |-- quantized --> Qwen/Qwen3-8B-FP8, Qwen/Qwen3-8B-AWQ, Qwen/Qwen3-8B-GGUF
 |                 |-- quantized --> unsloth/Qwen3-8B-GGUF, unsloth/Qwen3-8B-bnb-4bit
 |                 |-- finetune  --> mlabonne/Qwen3-8B-abliterated
 |                 |                   |-- quantized --> bartowski/mlabonne_Qwen3-8B-abliterated-GGUF
 |                 |                   |-- quantized --> mradermacher/Qwen3-8B-abliterated-GGUF
 |                 |                   |-- quantized --> warshanks/Qwen3-8B-abliterated-AWQ
 |                 |                   `-- finetune  --> braginpawel/qwen-3-8b-grpo
 |                 |-- finetune  --> huihui-ai/Huihui-Qwen3-8B-abliterated-v2
 |                 |                   `-- quantized --> mradermacher/Huihui-Qwen3-8B-abliterated-v2-GGUF
 |                 |-- finetune  --> Qwen/Qwen3Guard-Gen-8B       (safety classifier)
 |                 |-- adapter   --> don0ox1/Qwen3-8B-Roleplay-CN-v7  (PEFT/LoRA-style)
 |                 `-- merge     --> MergekitCloud/mergekit-105 (with DeepSeek-R1-0528-Qwen3-8B
 |                                    and ertghiu256/qwen3-8b-code-reasoning)
 |-- merge --> hon9kon9ize/CantoneseLLM-v2.0-8B-Thinking-Chat-Vector-Merged
 |             (also lists Qwen/Qwen3-8B as parent)
 `-- (not recorded) deepseek-ai/DeepSeek-R1-0528-Qwen3-8B   (distillation; no base_model tag)
```

Queried listings confirm `Qwen/Qwen3-8B` is a `finetune` child of `Qwen/Qwen3-8B-Base`, and the other relations above come from the metadata of each repo (`base_model` plus tags).

## 3. Edge by edge

| Edge                                                                                | Technique                                                                                                                                                                                    | Evidence            | See             |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------- |
| Base -> `Qwen/Qwen3-8B`                                                             | Qwen3 post-training. 8B is a lightweight model, so it uses strong-to-weak distillation (off-policy then on-policy) from Qwen3-32B/235B-A22B instead of the 4-stage pipeline                  | arXiv 2505.09388    | distillation.md |
| `Qwen3-8B` -> `-FP8`, `-AWQ`, `-GGUF`                                               | Post-training quantization (lower-precision weights, no change in behaviour intended). FP8, AWQ and GGUF are different formats                                                               | tags `quantized`    | n/a             |
| `Qwen3-8B` -> `mlabonne/Qwen3-8B-abliterated`                                       | Abliteration (refusal-direction removal). Card: "created with a new abliteration technique", a research project. HF relation is `finetune` even though no gradient training is the core step | card + tags         | abliteration.md |
| `Qwen3-8B` -> `huihui-ai/Huihui-Qwen3-8B-abliterated-v2`                            | Abliteration, tags `abliterated`, `uncensored`                                                                                                                                               | tags                | abliteration.md |
| abliterated -> GGUF/AWQ (bartowski, mradermacher, warshanks)                        | Quantization of the already-edited weights, so the quant inherits the edit                                                                                                                   | tags `quantized`    | n/a             |
| abliterated -> `braginpawel/qwen-3-8b-grpo`                                         | Further fine-tune created with Unsloth + TRL (card). The name suggests GRPO, but the card does not state the algorithm: UNVERIFIED                                                           | card                | rlvr.md         |
| `Qwen3-8B` -> `Qwen3Guard-Gen-8B`                                                   | Task fine-tune into a safety classifier                                                                                                                                                      | tags                | n/a             |
| `Qwen3-8B` -> `don0ox1/Qwen3-8B-Roleplay-CN-v7`                                     | Adapter (library_name `peft`, relation `adapter`)                                                                                                                                            | tags                | n/a             |
| Base + `Qwen3-8B` -> `hon9kon9ize/CantoneseLLM-v2.0-8B-Thinking-Chat-Vector-Merged` | Continued pre-training of Base on 784M tokens, then adding the chat vector (Qwen3-8B minus Base). Declared `merge`                                                                           | card (YAML + prose) | merging.md      |
| `Qwen3-8B` + 2 others -> `MergekitCloud/mergekit-105`                               | Mergekit-style merge, three declared parents. Method (SLERP/TIES/...) not read from the repo: UNVERIFIED                                                                                     | tags `merge`        | merging.md      |
| Base -> `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`                                     | SFT on R1-0528 chain-of-thought (distillation)                                                                                                                                               | card prose only     | distillation.md |

## 4. Teaching points for the website

- A tree edge says "derived from", not how. `finetune` mixes full SFT, DPO/RL, distillation, abliteration and any weight edit. Only `adapter`, `merge` and `quantized` are more specific.
- Edges compose: Base -> Instruct (lab) -> abliterated (community) -> GGUF (quantizer). A GGUF of an abliterated model is two edits deep.
- The parent can be wrong or absent, and there is no verification that the stated method was used.
- Qwen's instruct/thinking model is itself tagged a `finetune` of the Base, which is the same word as used for a hobbyist's edit.
- Qwen3-4B-Instruct-2507 (a separate 2025 release) is the Heretic example in abliteration.md, but it is not in this 8B tree, and HF's tree does not link it to its parent.

## Suggested visuals

1. Interactive tree: nodes colored by relation (finetune blue, quantized grey, adapter green, merge orange, "unrecorded" dashed). Click a node to see the technique, a one-line formula, and the repo id.
2. Download bars (snapshot 2026-11-09): Qwen3-8B 6.84M; Qwen3-8B-AWQ 1.62M; Qwen3-8B-GGUF 0.83M; Huihui-Qwen3-8B-abliterated-v2 20k; mlabonne/Qwen3-8B-abliterated 1.5k. Shows how fast usage drops away from the root.
3. Metadata diff panel: YAML of `mlabonne/Qwen3-8B-abliterated` (`base_model: [Qwen/Qwen3-8B]`) beside the tags HF derives from it.
4. "Missing edge" callout on DeepSeek-R1-0528-Qwen3-8B: card text versus empty tree.

## References

1. Qwen Team. "Qwen3 Technical Report." 2025. https://arxiv.org/abs/2505.09388
2. Hugging Face model pages / API: https://huggingface.co/Qwen/Qwen3-8B-Base ; https://huggingface.co/Qwen/Qwen3-8B ; https://huggingface.co/Qwen/Qwen3-8B-GGUF ; https://huggingface.co/mlabonne/Qwen3-8B-abliterated ; https://huggingface.co/huihui-ai/Huihui-Qwen3-8B-abliterated-v2 ; https://huggingface.co/bartowski/mlabonne_Qwen3-8B-abliterated-GGUF ; https://huggingface.co/braginpawel/qwen-3-8b-grpo ; https://huggingface.co/hon9kon9ize/CantoneseLLM-v2.0-8B-Thinking-Chat-Vector-Merged ; https://huggingface.co/MergekitCloud/mergekit-105 ; https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
3. Hugging Face Hub API, model listing by `base_model` filter: https://huggingface.co/api/models
