<script lang="ts">
	import 'katex/dist/katex.min.css';
	import '../app.css';
	import favicon from '#lib/assets/favicon.svg';
	import SiteHeader from '#lib/components/SiteHeader.svelte';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<SiteHeader />
<main>
	{@render children()}
</main>
<footer>
	<p>
		An interactive, simulated explainer of LLM post-training. Visuals use curated, illustrative
		data, not live models. Inspired by
		<a href="https://poloclub.github.io/transformer-explainer/">Transformer Explainer</a>.
	</p>
</footer>

<style>
	main {
		min-height: 70vh;
	}
	footer {
		margin-top: 96px;
		padding: 24px 16px 40px;
		border-top: 1px solid var(--line);
		text-align: center;
		font-size: 13px;
		color: var(--ink-3);
	}
</style>
