<script lang="ts">
	let {
		label,
		value = $bindable(),
		min,
		max,
		step = 0.01,
		format = (v: number) => String(v),
		hint = ''
	}: {
		label: string;
		value: number;
		min: number;
		max: number;
		step?: number;
		format?: (v: number) => string;
		hint?: string;
	} = $props();
</script>

<label class="slider" title={hint}>
	<span class="row"><span class="label">{label}</span><output>{format(value)}</output></span>
	<input type="range" bind:value {min} {max} {step} />
</label>

<style>
	.slider {
		display: flex;
		flex-direction: column;
		gap: 2px;
		min-width: 140px;
		flex: 1;
	}
	.row {
		display: flex;
		justify-content: space-between;
		font-size: 13px;
	}
	.label {
		color: var(--ink-2);
	}
	output {
		font-family: var(--mono);
		font-size: 12px;
	}
	input {
		width: 100%;
		accent-color: var(--ink);
	}
</style>
