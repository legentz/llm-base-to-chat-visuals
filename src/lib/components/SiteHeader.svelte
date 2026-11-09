<script lang="ts">
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { chapters, chapterHref } from '#lib/chapters.js';

	const current = $derived(page.url.pathname);
	const isActive = (slug: string) => current.includes(`/${slug}/`);
</script>

<header class="site-header">
	<div class="inner">
		<a class="brand" href={resolve('/')}>
			<span class="dot base"></span><span class="arrow">→</span><span class="dot instruct"></span>
			<span class="name">Post-Training Explainer</span>
		</a>
		<nav aria-label="Chapters">
			<ol>
				{#each chapters as c (c.slug)}
					<li>
						{#if c.ready}
							<a
								href={chapterHref(c.slug)}
								class:active={isActive(c.slug)}
								style:--stage={`var(--stage-${c.stage})`}
								aria-current={isActive(c.slug) ? 'page' : undefined}
							>
								<span class="n">{String(c.n).padStart(2, '0')}</span>{c.short}
							</a>
						{:else}
							<span class="soon" title="Coming soon">
								<span class="n">{String(c.n).padStart(2, '0')}</span>{c.short}
							</span>
						{/if}
					</li>
				{/each}
				<li><a href={resolve('references/')} class:active={isActive('references')}>Refs</a></li>
			</ol>
		</nav>
	</div>
</header>

<style>
	.site-header {
		position: sticky;
		top: 0;
		z-index: 20;
		background: color-mix(in srgb, var(--bg) 88%, transparent);
		backdrop-filter: blur(8px);
		border-bottom: 1px solid var(--line);
	}
	.inner {
		max-width: var(--maxw);
		margin: 0 auto;
		padding: 10px 16px;
		display: flex;
		align-items: center;
		gap: 20px;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 6px;
		color: var(--ink);
		text-decoration: none;
		font-weight: 700;
		white-space: nowrap;
	}
	.dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
	}
	.dot.base {
		background: var(--c-base);
	}
	.dot.instruct {
		background: var(--c-instruct);
	}
	.arrow {
		color: var(--ink-3);
		font-size: 12px;
	}
	.name {
		margin-left: 4px;
	}
	nav {
		overflow-x: auto;
		scrollbar-width: none;
		flex: 1;
	}
	ol {
		list-style: none;
		display: flex;
		gap: 4px;
		margin: 0;
		padding: 0;
	}
	nav a,
	.soon {
		display: inline-flex;
		align-items: baseline;
		gap: 4px;
		padding: 4px 10px;
		border-radius: 999px;
		font-size: 13px;
		white-space: nowrap;
		color: var(--ink-2);
		text-decoration: none;
	}
	nav a:hover {
		background: var(--surface-2);
	}
	nav a.active {
		background: var(--stage, var(--ink));
		color: #fff;
	}
	.soon {
		color: var(--ink-3);
		opacity: 0.6;
		cursor: default;
	}
	.n {
		font-family: var(--mono);
		font-size: 10px;
		opacity: 0.7;
	}
	@media (max-width: 640px) {
		.name {
			display: none;
		}
	}
</style>
