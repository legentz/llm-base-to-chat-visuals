<script lang="ts">
	import Heatmap from '#lib/components/Heatmap.svelte';
	import LossCurve from '#lib/components/LossCurve.svelte';
	import Slider from '#lib/components/Slider.svelte';
	import StepControls from '#lib/components/StepControls.svelte';
	import {
		loraParams,
		presetModules,
		qwen3_8b,
		scaling,
		totalParams,
		trainableParams,
		type Scaling,
		type TargetPreset
	} from '#lib/math/lora.js';
	import {
		bestRankError,
		learnedFractions,
		lowRankTarget,
		trainFactorization
	} from '#lib/math/lowrank.js';

	// ── Anatomy: one Qwen3-8B projection, W₀ + (α/r)·B·A ─────────────────────────────
	const RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
	const ALPHAS = [8, 16, 32, 64, 128];
	let rIdx = $state(4);
	let aIdx = $state(2);
	let scaleKind: Scaling = $state('lora');
	let preset: TargetPreset = $state('all-linear');
	let selected = $state('q_proj');

	const r = $derived(RANKS[rIdx]);
	const alpha = $derived(ALPHAS[aIdx]);
	const gamma = $derived(scaling(alpha, r, scaleKind));
	const shape = $derived(qwen3_8b.linears.find((l) => l.name === selected)!);
	const targeted = $derived(presetModules[preset].includes(selected));
	const total = totalParams(qwen3_8b);
	const trainable = $derived(trainableParams(qwen3_8b, r, presetModules[preset]));

	// Drawn to scale: the larger side of the selected matrix is 190 px. B and A get at least 1.5 px.
	const S = $derived(190 / Math.max(shape.dIn, shape.dOut));
	const w0 = $derived({ w: shape.dIn * S, h: shape.dOut * S });
	const thin = $derived(Math.max(1.5, r * S));
	const bx = $derived(20 + w0.w + 96);
	const ax = $derived(bx + thin + 18);

	const fmt = (x: number) =>
		x >= 1e9
			? `${(x / 1e9).toFixed(2)}B`
			: x >= 1e6
				? `${(x / 1e6).toFixed(1)}M`
				: x >= 1e3
					? `${(x / 1e3).toFixed(1)}k`
					: String(x);
	const pct = (x: number) => (x < 0.001 ? (x * 100).toFixed(3) : (x * 100).toFixed(2)) + '%';

	// ── Toy training: fit ΔW = BA to a 24×24 target update by gradient descent ─────────
	const N = 24;
	const STEPS = 300;
	const TOY_RANKS = [1, 2, 4, 8, 16, 24];
	const target = lowRankTarget(
		N,
		Array.from({ length: N }, (_, i) => 3 * 0.8 ** i),
		5
	);
	let toyIdx = $state(2);
	let step = $state(0);
	const toyR = $derived(TOY_RANKS[toyIdx]);
	const traj = $derived(trainFactorization(target, toyR, STEPS, 0.035));
	const dW = $derived(traj.snapshots[step]);
	const residual = $derived(Float32Array.from(target.T, (t, i) => t - dW[i]));
	const fractions = $derived(learnedFractions(target, dW));
	const best = $derived(bestRankError(target.sigmas, toyR));
	const vmax = Math.max(...Array.from(target.T, Math.abs)) * 0.8;
</script>

<div class="lab">
	<div class="card anatomy">
		<div class="controls">
			<Slider
				label="rank r"
				bind:value={rIdx}
				min={0}
				max={RANKS.length - 1}
				step={1}
				format={(i) => String(RANKS[i])}
			/>
			<Slider
				label="lora_alpha α"
				bind:value={aIdx}
				min={0}
				max={ALPHAS.length - 1}
				step={1}
				format={(i) => String(ALPHAS[i])}
			/>
			<div class="group" role="group" aria-label="Scaling">
				<span class="eyebrow">Scale</span>
				<button class="btn" aria-pressed={scaleKind === 'lora'} onclick={() => (scaleKind = 'lora')}
					>α / r</button
				>
				<button
					class="btn"
					aria-pressed={scaleKind === 'rslora'}
					onclick={() => (scaleKind = 'rslora')}>α / √r (rsLoRA)</button
				>
			</div>
		</div>

		<div class="group modules" role="group" aria-label="Matrix to draw">
			<span class="eyebrow">Matrix</span>
			{#each qwen3_8b.linears as l (l.name)}
				<button
					class="btn mod"
					class:on={presetModules[preset].includes(l.name)}
					aria-pressed={selected === l.name}
					onclick={() => (selected = l.name)}>{l.name}</button
				>
			{/each}
		</div>

		<svg
			viewBox={`0 0 540 ${w0.h + 60}`}
			class="anat"
			role="img"
			aria-label={`${selected}: frozen ${shape.dOut}×${shape.dIn} weight plus a rank-${r} update B·A`}
		>
			<g transform="translate(0,34)">
				<text x="20" y="-12" class="lab-t">W₀ · frozen</text>
				<rect x="20" width={w0.w} height={w0.h} class="w0" />
				<text x={20 + w0.w / 2} y={w0.h + 18} class="dim" text-anchor="middle"
					>{shape.dOut} × {shape.dIn} = {fmt(shape.dIn * shape.dOut)}</text
				>
				<text x={20 + w0.w + 14} y={Math.min(w0.h / 2, 60) + 6} class="op">+</text>
				<text x={20 + w0.w + 30} y={Math.min(w0.h / 2, 60) + 5} class="gamma"
					>{gamma.toFixed(gamma < 1 ? 3 : 2)}·</text
				>

				<g class:off={!targeted}>
					<text x={bx} y={w0.h + 18} class="lab-t b">B (init 0)</text>
					<rect x={bx} width={thin} height={w0.h} class="b" />
					<text x={ax} y="-12" class="lab-t a">A (random init)</text>
					<rect x={ax} width={w0.w} height={thin} class="a" />
				</g>
			</g>
		</svg>
		<p class="sizes" class:off={!targeted}>
			<span class="dimtxt"
				>B {shape.dOut}×{r} + A {r}×{shape.dIn} = {fmt(loraParams(r, shape))} params</span
			>
			<strong>{pct(loraParams(r, shape) / (shape.dIn * shape.dOut))} of this matrix</strong>
			<span class="dimtxt"
				>· drawn to scale{#if !targeted}
					· not adapted with this target_modules preset{/if}</span
			>
		</p>

		<div class="group" role="group" aria-label="Target modules">
			<span class="eyebrow">target_modules</span>
			<button class="btn" aria-pressed={preset === 'qv'} onclick={() => (preset = 'qv')}
				>q, v (2021 paper)</button
			>
			<button class="btn" aria-pressed={preset === 'attn'} onclick={() => (preset = 'attn')}
				>attention</button
			>
			<button
				class="btn"
				aria-pressed={preset === 'all-linear'}
				onclick={() => (preset = 'all-linear')}>all-linear</button
			>
		</div>

		<dl class="stats">
			<div>
				<dt>trainable (whole model)</dt>
				<dd>{fmt(trainable)}</dd>
			</div>
			<div>
				<dt>share of Qwen3-8B</dt>
				<dd>{pct(trainable / total)}</dd>
			</div>
			<div>
				<dt>adapter file (bf16)</dt>
				<dd>{((trainable * 2) / 1e6).toFixed(0)} MB</dd>
			</div>
			<div>
				<dt>full checkpoint (bf16)</dt>
				<dd>{((total * 2) / 1e9).toFixed(1)} GB</dd>
			</div>
		</dl>
	</div>

	<div class="card toy">
		<div class="toyhead">
			<p class="eyebrow">Training a rank-{toyR} update on a toy 24 × 24 matrix</p>
			<Slider
				label="toy rank"
				bind:value={toyIdx}
				min={0}
				max={TOY_RANKS.length - 1}
				step={1}
				format={(i) => String(TOY_RANKS[i])}
			/>
		</div>
		<StepControls bind:step max={STEPS} interval={35} label="Step" />
		<div class="maps">
			<Heatmap values={target.T} n={N} max={vmax} label="target ΔW* (full fine-tune)" />
			<Heatmap values={dW} n={N} max={vmax} label={`LoRA ΔW = BA at step ${step}`} />
			<Heatmap values={residual} n={N} max={vmax} label="what's still missing" />
		</div>
		<div class="toygrid">
			<div>
				<p class="eyebrow">Learned share of each target direction</p>
				<svg
					viewBox="0 0 240 90"
					class="dirs"
					role="img"
					aria-label="Learned fraction per direction"
				>
					<rect
						x={10 + toyR * 9.5}
						y="0"
						width={Math.max(0, 240 - 10 - toyR * 9.5)}
						height="76"
						class="unreach"
					/>
					{#each fractions as f, k (k)}
						<rect x={10 + k * 9.5} y="4" width="7" height="68" class="track" />
						<rect
							x={10 + k * 9.5}
							y={4 + 68 * (1 - Math.max(0, Math.min(1, f)))}
							width="7"
							height={68 * Math.max(0, Math.min(1, f))}
							class="fill"
							style:opacity={0.35 + 0.65 * Math.sqrt(target.sigmas[k] / target.sigmas[0])}
						/>
					{/each}
					<text x="10" y="88" class="tick">σ₁ (largest)</text>
					<text x="236" y="88" class="tick" text-anchor="end">σ₂₄</text>
				</svg>
				<p class="hint">Shaded: directions a rank-{toyR} update can't hold.</p>
			</div>
			<div>
				<LossCurve
					series={[
						{ name: 'relative error', color: 'var(--c-lora)', values: traj.relError },
						{
							name: `best possible at rank ${toyR}`,
							color: 'var(--ink-3)',
							values: traj.relError.map(() => best),
							dashed: true
						}
					]}
					{step}
					yMax={1}
					yLabel="‖ΔW* − ΔW‖² / ‖ΔW*‖²"
					height={150}
				/>
			</div>
		</div>
	</div>

	<p class="illustrative">
		Top: real Qwen3-8B shapes (config.json), exact parameter counts. Bottom: a live toy, real
		gradient descent on ½‖BA − ΔW*‖² from B = 0, with a synthetic target whose singular values decay
		geometrically. It shows the mechanics, not the behaviour of any trained adapter.
	</p>
</div>

<style>
	.lab {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.anatomy,
	.toy {
		padding: 14px 16px;
		display: flex;
		flex-direction: column;
		gap: 10px;
		min-width: 0;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 22px;
		align-items: flex-end;
	}
	.group {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.group .eyebrow {
		margin: 0 4px 0 0;
	}
	.mod {
		font-family: var(--mono);
		font-size: 12px;
	}
	.mod.on:not([aria-pressed='true']) {
		border-color: var(--c-lora);
	}
	svg {
		width: 100%;
		height: auto;
		font-family: var(--font);
	}
	.anat {
		max-width: 600px;
	}
	.sizes {
		margin: -4px 0 0;
		font-family: var(--mono);
		font-size: 12.5px;
		display: flex;
		flex-wrap: wrap;
		gap: 2px 8px;
	}
	.sizes.off {
		opacity: 0.5;
	}
	.dimtxt {
		color: var(--ink-3);
	}
	.w0 {
		fill: var(--c-masked);
		fill-opacity: 0.5;
		stroke: var(--ink-3);
	}
	.b,
	.a {
		fill: var(--c-lora);
		transition:
			width 0.25s,
			height 0.25s;
	}
	.off {
		opacity: 0.25;
	}
	.lab-t {
		font-size: 14px;
		font-weight: 600;
		fill: var(--ink-2);
	}
	.lab-t.a,
	.lab-t.b {
		fill: var(--c-lora);
	}
	.dim {
		font-size: 12.5px;
		font-family: var(--mono);
		fill: var(--ink-3);
	}
	.op {
		font-size: 22px;
		fill: var(--ink-2);
	}
	.gamma {
		font-size: 13px;
		font-family: var(--mono);
		fill: var(--ink);
	}
	.stats {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 8px;
		margin: 0;
	}
	@media (max-width: 640px) {
		.stats {
			grid-template-columns: repeat(2, 1fr);
		}
	}
	.stats div {
		background: var(--surface-2);
		border-radius: 8px;
		padding: 8px 10px;
	}
	dt {
		font-size: 11.5px;
		color: var(--ink-3);
	}
	dd {
		margin: 2px 0 0;
		font-family: var(--mono);
		font-size: 17px;
		font-weight: 600;
	}
	.toyhead {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: flex-end;
		gap: 10px;
	}
	.toyhead .eyebrow {
		margin: 0;
	}
	.toyhead :global(.slider) {
		max-width: 220px;
	}
	.maps {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 10px;
		justify-items: center;
	}
	.toygrid {
		display: grid;
		grid-template-columns: 1fr 1.4fr;
		gap: 16px;
		align-items: start;
	}
	@media (max-width: 720px) {
		.toygrid {
			grid-template-columns: 1fr;
		}
	}
	.toygrid .eyebrow {
		margin: 0 0 4px;
	}
	.dirs {
		max-width: 360px;
	}
	.unreach {
		fill: var(--c-masked);
		fill-opacity: 0.3;
	}
	.track {
		fill: var(--line);
		fill-opacity: 0.5;
	}
	.fill {
		fill: var(--c-lora);
	}
	.tick {
		font-size: 9px;
		fill: var(--ink-3);
	}
	.hint {
		margin: 2px 0 0;
		font-size: 12px;
		color: var(--ink-3);
	}
	.illustrative {
		margin: 0;
	}
</style>
