<script lang="ts">
	import katex from 'katex';

	/** Renders LaTeX with KaTeX at prerender time; `display` gives a centred block formula. */
	let { expr, display = false }: { expr: string; display?: boolean } = $props();
	const html = $derived(katex.renderToString(expr, { displayMode: display, throwOnError: true }));
</script>

{#if display}
	<!-- eslint-disable-next-line svelte/no-at-html-tags -- KaTeX output from our own LaTeX strings -->
	<div class="tex-block">{@html html}</div>
{:else}
	<!-- eslint-disable-next-line svelte/no-at-html-tags -- KaTeX output from our own LaTeX strings -->
	<span>{@html html}</span>
{/if}

<style>
	.tex-block {
		margin: 1.2em 0;
		padding: 10px 16px;
		background: var(--surface-2);
		border-radius: 8px;
		overflow-x: auto;
	}
</style>
