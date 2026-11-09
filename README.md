# LLM Base-to-Chat Visuals

**Live site:** https://legentz.github.io/llm-base-to-chat-visuals/

An interactive, **simulated** explainer of how a base (pretrained) LLM becomes a chat / instruct
model, and how community variants on Hugging Face (LoRA fine-tunes, abliterated models, merges,
distilled models) are produced. It's inspired by
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/).

Nothing is trained live. Every visual is driven by curated, illustrative data plus small pieces of
real math computed in the browser (losses, advantages, projections). Claims are cited to primary
sources on the References page.

## Chapters

1. Base vs Instruct · 2. Chat templates & SFT · 3. LoRA & QLoRA · 4. Reward models & RLHF ·
2. DPO & friends · 6. RLVR & GRPO · 7. Distillation · 8. Abliteration · 9. Model merging ·
3. A Hugging Face family tree

## Run locally

```sh
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests for the math
npm run build      # static site in ./build
```

Contributor notes (architecture, conventions, invariants) are in [CLAUDE.md](CLAUDE.md).
Research notes with sources for each chapter are in [`docs/research/`](docs/research/).
