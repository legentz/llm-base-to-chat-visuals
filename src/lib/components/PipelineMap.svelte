<script lang="ts">
	import { chapters, chapterHref } from '#lib/chapters.js';

	/**
	 * Landing-page map: model checkpoints are nodes, techniques are edges (the same shape as a
	 * Hugging Face "model tree"). Edge pills link to the chapter that explains the technique.
	 */
	interface Node {
		id: string;
		x: number;
		y: number;
		title: string;
		sub: string;
		color: string;
	}
	interface Edge {
		from: string;
		to: string;
		label: string;
		chapter: string;
		/** Optional label offset along the curve, as a fraction (default 0.5). */
		t?: number;
	}

	const W = 156;
	const H = 58;
	const nodes: Node[] = [
		{
			id: 'corpus',
			x: 85,
			y: 90,
			title: 'Web-scale text',
			sub: '10T+ tokens',
			color: 'var(--ink-3)'
		},
		{
			id: 'base',
			x: 365,
			y: 90,
			title: 'Base model',
			sub: 'Qwen3-8B-Base',
			color: 'var(--c-base)'
		},
		{
			id: 'sft',
			x: 645,
			y: 90,
			title: 'SFT model',
			sub: 'chat format',
			color: 'var(--stage-alignment)'
		},
		{
			id: 'pref',
			x: 925,
			y: 90,
			title: 'Preference-tuned',
			sub: 'helpful, harmless',
			color: 'var(--stage-alignment)'
		},
		{
			id: 'instruct',
			x: 1185,
			y: 90,
			title: 'Chat model',
			sub: 'Qwen3-8B',
			color: 'var(--c-instruct)'
		},
		{
			id: 'student',
			x: 365,
			y: 320,
			title: 'Small student',
			sub: 'R1-distilled',
			color: 'var(--stage-efficiency)'
		},
		{
			id: 'abliterated',
			x: 645,
			y: 320,
			title: 'Abliterated',
			sub: '…-abliterated',
			color: 'var(--stage-derivation)'
		},
		{
			id: 'lora',
			x: 925,
			y: 320,
			title: 'Domain fine-tune',
			sub: 'LoRA adapter',
			color: 'var(--stage-efficiency)'
		},
		{
			id: 'merge',
			x: 1185,
			y: 320,
			title: 'Merged model',
			sub: 'mergekit',
			color: 'var(--stage-derivation)'
		}
	];
	const edges: Edge[] = [
		{ from: 'corpus', to: 'base', label: 'Pretraining', chapter: 'base-vs-instruct' },
		{ from: 'base', to: 'sft', label: 'SFT', chapter: 'sft' },
		{ from: 'sft', to: 'pref', label: 'RLHF · DPO', chapter: 'rlhf' },
		{ from: 'pref', to: 'instruct', label: 'RLVR', chapter: 'rlvr' },
		{ from: 'instruct', to: 'student', label: 'Distillation', chapter: 'distillation', t: 0.6 },
		{ from: 'instruct', to: 'abliterated', label: 'Abliteration', chapter: 'abliteration', t: 0.6 },
		{ from: 'instruct', to: 'lora', label: 'LoRA', chapter: 'lora', t: 0.6 },
		{ from: 'instruct', to: 'merge', label: 'Merging', chapter: 'merging', t: 0.45 },
		{ from: 'lora', to: 'merge', label: '', chapter: 'merging' }
	];

	const byId = new Map(nodes.map((n) => [n.id, n]));
	const chapterBySlug = new Map(chapters.map((c) => [c.slug, c]));

	/** Cubic path between node borders: horizontal for same-row nodes, vertical-ish otherwise. */
	function geometry(e: Edge) {
		const a = byId.get(e.from)!;
		const b = byId.get(e.to)!;
		let p0: [number, number], p3: [number, number], c1: [number, number], c2: [number, number];
		if (a.y === b.y) {
			const dir = Math.sign(b.x - a.x);
			p0 = [a.x + (dir * W) / 2, a.y];
			p3 = [b.x - (dir * W) / 2, b.y];
			c1 = [(p0[0] + p3[0]) / 2, p0[1]];
			c2 = c1;
		} else {
			p0 = [a.x, a.y + H / 2];
			p3 = [b.x, b.y - H / 2];
			c1 = [p0[0], p0[1] + 110];
			c2 = [p3[0], p3[1] - 110];
		}
		const t = e.t ?? 0.5;
		const u = 1 - t;
		const pt = (i: 0 | 1) =>
			u ** 3 * p0[i] + 3 * u * u * t * c1[i] + 3 * u * t * t * c2[i] + t ** 3 * p3[i];
		return {
			d: `M${p0[0]},${p0[1]} C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${p3[0]},${p3[1]}`,
			lx: pt(0),
			ly: pt(1)
		};
	}

	let hover: string | null = $state(null);
	const active = (e: Edge) => hover === e.chapter;
</script>

<figure class="map card">
	<svg viewBox="0 0 1280 400" role="img" aria-labelledby="map-title">
		<title id="map-title">
			Post-training pipeline: base model to chat model, plus community derivations
		</title>
		<defs>
			<marker
				id="arrow"
				viewBox="0 0 10 10"
				refX="9"
				refY="5"
				markerWidth="7"
				markerHeight="7"
				orient="auto-start-reverse"
			>
				<path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
			</marker>
		</defs>

		<text class="row-label" x="10" y="30">Core pipeline (what the lab does)</text>
		<text class="row-label" x="10" y="390">Derivations (what the community does)</text>
		<line x1="10" x2="1270" y1="205" y2="205" class="divider" />

		{#each edges as e (e.from + e.to)}
			{@const g = geometry(e)}
			<path d={g.d} class="edge" class:active={active(e)} marker-end="url(#arrow)" />
			<path d={g.d} class="flow" class:active={active(e)} />
		{/each}

		{#each nodes as n (n.id)}
			<g transform={`translate(${n.x - W / 2},${n.y - H / 2})`}>
				<rect width={W} height={H} rx="10" class="node" style:--nc={n.color} />
				<rect width="6" height={H} rx="3" fill={n.color} />
				<text x="16" y="25" class="ntitle">{n.title}</text>
				<text x="16" y="43" class="nsub">{n.sub}</text>
			</g>
		{/each}

		{#each edges.filter((e) => e.label) as e (e.label)}
			{@const g = geometry(e)}
			{@const ch = chapterBySlug.get(e.chapter)!}
			{@const w = e.label.length * 7 + 38}
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<g
				transform={`translate(${g.lx - w / 2},${g.ly - 13})`}
				onmouseenter={() => (hover = e.chapter)}
				onmouseleave={() => (hover = null)}
			>
				{#if ch.ready}
					<a href={chapterHref(ch.slug)} aria-label={`Chapter ${ch.n}: ${ch.title}`}>
						<rect width={w} height="26" rx="13" class="pill ready" class:active={active(e)} />
						<text x="12" y="17" class="pnum">{String(ch.n).padStart(2, '0')}</text>
						<text x="30" y="17" class="plabel">{e.label}</text>
					</a>
				{:else}
					<rect width={w} height="26" rx="13" class="pill" />
					<text x="12" y="17" class="pnum">{String(ch.n).padStart(2, '0')}</text>
					<text x="30" y="17" class="plabel soon">{e.label}</text>
				{/if}
			</g>
		{/each}
	</svg>
	<figcaption>
		Nodes are checkpoints; arrows are techniques. Each arrow is also an edge in a Hugging Face model
		tree (<code>base_model</code> → fine-tune / adapter / merge). Click a technique to open its chapter.
		LoRA adapters and abliteration can be applied at any stage; the map shows the common case.
	</figcaption>
</figure>

<style>
	.map {
		margin: 0;
		padding: 12px;
		overflow-x: auto;
	}
	svg {
		display: block;
		width: 100%;
		min-width: 640px;
		height: auto;
		font-family: var(--font);
	}
	.row-label {
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		fill: var(--ink-3);
	}
	.divider {
		stroke: var(--line);
		stroke-dasharray: 4 4;
	}
	.node {
		fill: var(--surface);
		stroke: var(--line);
	}
	.ntitle {
		font-size: 14px;
		font-weight: 600;
		fill: var(--ink);
	}
	.nsub {
		font-size: 11.5px;
		fill: var(--ink-3);
		font-family: var(--mono);
	}
	.edge {
		fill: none;
		stroke: var(--line);
		stroke-width: 2;
		transition: stroke 0.2s;
	}
	.edge.active {
		stroke: var(--ink-2);
	}
	.flow {
		fill: none;
		stroke: var(--c-instruct);
		stroke-width: 2;
		stroke-dasharray: 3 14;
		opacity: 0.55;
		animation: flow 1.6s linear infinite;
	}
	.flow.active {
		opacity: 1;
	}
	@keyframes flow {
		to {
			stroke-dashoffset: -34;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.flow {
			animation: none;
		}
	}
	.pill {
		fill: var(--surface-2);
		stroke: var(--line);
	}
	.pill.ready {
		fill: var(--surface);
		stroke: var(--ink-3);
		cursor: pointer;
	}
	.pill.active {
		fill: var(--ink);
	}
	.pill.active ~ text,
	a:hover .plabel,
	a:hover .pnum {
		fill: var(--bg);
	}
	a:hover .pill {
		fill: var(--ink);
	}
	.pnum {
		font-size: 10px;
		font-family: var(--mono);
		fill: var(--ink-3);
	}
	.plabel {
		font-size: 12.5px;
		font-weight: 600;
		fill: var(--ink);
	}
	.plabel.soon {
		fill: var(--ink-3);
	}
	figcaption {
		margin-top: 8px;
		font-size: 13px;
		color: var(--ink-3);
	}
</style>
