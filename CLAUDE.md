# CLAUDE.md: LLM Base-to-Chat Visuals (llm-base-to-chat-visuals)

Interactive, simulated explainer (in the style of Transformer Explainer) of LLM post-training: base → SFT → preference
optimisation → RLVR, plus LoRA, distillation, abliteration and merging. What the product is: see README.md.

## Commands

- `npm run dev` · `npm run build` (static → `build/`) · `npm run preview`
- Before every commit: `npm run format && npm run lint && npm run check && npm test && npm run build`
  (CI in `.github/workflows/ci.yml` runs the same, minus format).
- Build as GitHub Pages does it: `BASE_PATH=/llm-base-to-chat-visuals npm run build`.

## Stack & layout

SvelteKit 3 + Svelte 5 (runes forced), Vite 8, adapter-static (all pages prerendered, `trailingSlash: 'always'`),
D3 v7, KaTeX (rendered at prerender time), Vitest. SvelteKit config lives in `vite.config.ts` (no svelte.config.js in Kit 3).

- `src/lib/chapters.ts`: chapter registry (order, stage, `ready` flag). It drives the nav, the landing page and prev/next.
  Set `ready: true` when a chapter's route lands.
- `src/routes/<slug>/+page.svelte`: one folder per chapter, wrapped in `ChapterShell` (title, `hero` snippet = the
  interactive visual, children = article).
- `src/lib/components/`: shared visuals and helpers (`Cite`, `Tex`, `ChapterShell`, `SiteHeader`, …).
- `src/lib/math/`: pure, unit-tested functions (`*.spec.ts` next to them). All the numbers the visuals show come from here.
- `src/lib/data/`: curated illustrative data + `references.ts` (single bibliography).
- `docs/research/<chapter>.md`: verified notes with sources, written before the chapter is built. Items marked
  UNVERIFIED must not be presented as fact on the site.

## Domain invariants (if these break you get plausible-looking but false numbers, with no error)

- Data is **illustrative**, not recorded model output. The UI must say so (`.illustrative` class) wherever we show
  "model outputs" or training curves.
- SFT loss averages cross-entropy over **assistant tokens only** (mask = 0 on system/user/template tokens).
- DPO: implicit reward r = β·log(πθ/πref); a larger β keeps the policy closer to the reference. Loss = −log σ(margin).
- GRPO advantage = (r_i − mean(r)) / std(r) **within a group** of samples of the same prompt (Dr. GRPO drops the std).
- LoRA: ΔW = (α/r)·B·A, B initialised to 0, so ΔW = 0 at step 0. Trainable params per matrix = r·(d_in + d_out).
  Qwen3-8B shapes in `lora.ts` come from its config.json (8,190,735,360 params, untied embeddings; pinned in tests).
  Memory excludes activations: full FT 12 B/param (16 with fp32 master); adapters 16 B/param (fp32 w+grad+Adam);
  QLoRA = NF4 (4.127 bit with double quant) on block linears only, embeddings/lm_head stay bf16.
- Qwen3 template: the final assistant turn of a _training_ example always carries `<think>\n…\n</think>\n\n`
  (empty for non-thinking samples); `enable_thinking` only changes the _generation prompt_. Llama 3.1 always
  emits a system turn with "Cutting Knowledge Date / Today Date" lines. Both are pinned in `templates.spec.ts`.
- Abliteration: direction r̂ = normalised diff-of-means(harmful, harmless) at a chosen layer; ablation x' = x − r̂r̂ᵀx.
  Educational only: explain the geometry and trade-offs. No runnable recipes for real models.

## Conventions

- Site text, code, comments and commits in **English**. Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
- One branch per major change (`feat/<chapter>`, `chore/…`, `docs/…`), merged into `main` via PR once CI is green.
- Every factual claim on the site gets a `<Cite id="…">`; `references.spec.ts` fails if an id is missing from
  `references.ts`. Prefer primary sources (arXiv abs, tech reports, official blogs, HF model cards).
- Colours are semantic tokens in `src/app.css` (`--c-base`, `--c-instruct`, `--c-chosen`, `--c-rejected`, …): reuse
  them, never hard-code hex in components. Light + dark both supported.
- Prettier: tabs, single quotes, width 100.

## Working agreement

- Workflow per chapter: research note (cheap agent) → build → SOTA review by a separate agent → link check → PR.
- Never commit to `main` directly; never force-push `main`. Each branch → `gh pr create` → wait for CI
  (`gh pr checks --watch`). **The agent merges (`gh pr merge --merge`) only if CI is green, there are no conflicts
  and nothing regressed** (full local suite + visual check of existing chapters); otherwise it hands the PR link +
  summary to the user, who reviews and merges. Start the next branch from `main` once the previous PR is merged; if it isn't
  merged yet, branch from the previous branch and say so in the PR body.
- Repo: https://github.com/legentz/llm-base-to-chat-visuals · live: https://legentz.github.io/llm-base-to-chat-visuals/
  (Pages deploys from `main` via `.github/workflows/deploy.yml`).
- Visual check without the Chrome extension: `npx http-server@14.1.1 build -p 4174 -s` then headless Chrome
  `--screenshot`. Headless Chrome won't go below 500px wide, so test 390px mobile through a same-origin wrapper
  (`build/_mobile.html` with a 390px iframe). `vite preview` dies when `build/` is rebuilt under it.

## Traps already hit

- Kit 3 uses the `#lib` alias (package.json `imports`), not `$lib`. TS imports need the `.js` extension
  (`'#lib/chapters.js'`); `.svelte` imports don't.
- `resolve()` from `$app/paths` takes paths **without** a leading slash for pathnames (`resolve('references/')`);
  chapter links go through `chapterHref(slug)`. There's no `base` export anymore. The ESLint rule
  `svelte/no-navigation-without-resolve` is off because it can't see through the helpers.
- Illustrative token data (`generations.ts`) uses `tok:p|tok:p*` steps: the splitter ignores `|` inside `<|…|>`,
  and a literal `:` token is written `::0.74`.
- Sampler filters run sequentially on the renormalised survivors (top-k → top-p → min-p), as HF does.
- Prettier reflows long lines in `.svelte` (even splitting `<code>` tags), so scripted string replacement on
  page text silently misses. Edit page prose with the Edit tool against the current file.
- Headless screenshots with a very tall window (≳2000px) of the 390px iframe wrapper render shifted/cropped on the
  left: an artifact, not a layout bug. Check overflow via DOM (`scrollWidth`) or screenshot ≤1300px tall.
- `paths.relative: true` → built HTML uses `./`-relative links, so the same build works at any base path.
