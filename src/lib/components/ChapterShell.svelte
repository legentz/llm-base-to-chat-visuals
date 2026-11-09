<script lang="ts">
	import type { Snippet } from 'svelte';
	import { chapters, chapterHref, neighbours, stageLabel } from '#lib/chapters.js';

	/**
	 * Common frame for every chapter: title block, full-width interactive hero, long-form article,
	 * prev/next navigation. Pages pass the hero as a snippet and the article as children.
	 */
	let { slug, hero, children }: { slug: string; hero: Snippet; children: Snippet } = $props();

	const chapter = $derived(chapters.find((c) => c.slug === slug)!);
	const nav = $derived(neighbours(slug));
</script>

<svelte:head>
	<title>{chapter.title} · Post-Training Explainer</title>
	<meta name="description" content={chapter.blurb} />
</svelte:head>

<section class="title" style:--stage={`var(--stage-${chapter.stage})`}>
	<p class="eyebrow">
		<span class="pill">Chapter {String(chapter.n).padStart(2, '0')}</span>
		{stageLabel[chapter.stage]}
	</p>
	<h1>{chapter.title}</h1>
	<p class="blurb">{chapter.blurb}</p>
</section>

<section class="hero">
	{@render hero()}
</section>

<article class="prose">
	{@render children()}
</article>

<nav class="pager prose" aria-label="Chapter navigation">
	{#if nav.prev}
		<a href={chapterHref(nav.prev.slug)} class="prev">
			<span class="eyebrow">← Previous</span>{nav.prev.title}
		</a>
	{:else}
		<span></span>
	{/if}
	{#if nav.next}
		<a href={chapterHref(nav.next.slug)} class="next">
			<span class="eyebrow">Next →</span>{nav.next.title}
		</a>
	{/if}
</nav>

<style>
	.title {
		max-width: var(--maxw);
		margin: 0 auto;
		padding: 40px 16px 8px;
	}
	.pill {
		display: inline-block;
		margin-right: 8px;
		padding: 2px 8px;
		border-radius: 999px;
		background: var(--stage);
		color: #fff;
	}
	h1 {
		margin: 10px 0 6px;
		font-size: clamp(1.9rem, 4vw, 2.8rem);
	}
	.blurb {
		margin: 0;
		font-size: 1.15rem;
		color: var(--ink-2);
	}
	.hero {
		max-width: var(--maxw);
		margin: 24px auto 56px;
		padding: 0 16px;
	}
	.pager {
		display: flex;
		justify-content: space-between;
		gap: 16px;
		margin-top: 64px;
	}
	.pager a {
		display: flex;
		flex-direction: column;
		padding: 12px 16px;
		border: 1px solid var(--line);
		border-radius: var(--radius);
		text-decoration: none;
		color: var(--ink);
		font-weight: 600;
	}
	.pager a:hover {
		border-color: var(--ink-3);
	}
	.next {
		text-align: right;
		margin-left: auto;
	}
</style>
