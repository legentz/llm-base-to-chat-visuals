<script lang="ts">
	import { onDestroy } from 'svelte';

	/** Play / pause / step / reset over an integer step in [0, max]. Shared by every scrubbed visual. */
	let {
		step = $bindable(0),
		max,
		interval = 450,
		label = 'Step'
	}: { step: number; max: number; interval?: number; label?: string } = $props();

	let playing = $state(false);
	let timer: ReturnType<typeof setInterval> | undefined;

	function stop() {
		playing = false;
		clearInterval(timer);
	}
	function play() {
		if (step >= max) step = 0;
		playing = true;
		timer = setInterval(() => {
			if (step >= max) stop();
			else step += 1;
		}, interval);
	}
	onDestroy(stop);
</script>

<div class="controls">
	<button
		class="btn"
		onclick={() => (playing ? stop() : play())}
		aria-label={playing ? 'Pause' : 'Play'}
	>
		{playing ? '❚❚ Pause' : '▶ Play'}
	</button>
	<button
		class="btn"
		onclick={() => (stop(), (step = Math.max(0, step - 1)))}
		aria-label="Step back">‹</button
	>
	<button
		class="btn"
		onclick={() => (stop(), (step = Math.min(max, step + 1)))}
		aria-label="Step forward">›</button
	>
	<button class="btn" onclick={() => (stop(), (step = 0))}>Reset</button>
	<label class="scrub">
		<span class="sr-only">{label}</span>
		<input type="range" min="0" {max} bind:value={step} oninput={stop} />
	</label>
	<span class="count mono">{label} {step}/{max}</span>
</div>

<style>
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.scrub {
		flex: 1;
		min-width: 120px;
	}
	.scrub input {
		width: 100%;
		accent-color: var(--ink);
	}
	.count {
		font-size: 12px;
		color: var(--ink-3);
		min-width: 80px;
		text-align: right;
	}
</style>
