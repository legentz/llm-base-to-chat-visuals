<script lang="ts">
	import { references } from '#lib/data/references.js';

	const sorted = [...references].sort((a, b) => a.cite.localeCompare(b.cite));
</script>

<svelte:head>
	<title>References · Post-Training Explainer</title>
</svelte:head>

<div class="prose">
	<p class="eyebrow">Bibliography</p>
	<h1>References</h1>
	<p>
		Every claim in the chapters links here. Sources are primary where possible: papers, technical
		reports, official blog posts and Hugging Face model cards.
	</p>
	<ol class="refs">
		{#each sorted as r (r.id)}
			<li id={r.id}>
				<span class="authors">{r.authors}</span>
				<span class="year">({r.year}).</span>
				<a href={r.url} rel="external noopener">{r.title}</a>{#if r.venue}.
					<span class="venue">{r.venue}</span>{/if}
			</li>
		{/each}
	</ol>
</div>

<style>
	h1 {
		font-size: 2.2rem;
		margin: 0.2em 0 0.4em;
	}
	.prose {
		padding-top: 48px;
	}
	.refs {
		padding-left: 1.2em;
	}
	.refs li {
		margin: 0.8em 0;
		scroll-margin-top: 80px;
	}
	.refs li:target {
		background: var(--surface-2);
		border-radius: 6px;
	}
	.authors {
		color: var(--ink);
	}
	.venue {
		font-style: italic;
	}
</style>
