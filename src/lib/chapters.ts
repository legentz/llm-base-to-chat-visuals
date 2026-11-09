/**
 * Single registry of chapters: drives the top navigation, the pipeline map on the landing page
 * and prev/next links. Order = reading order.
 */
import { resolve } from '$app/paths';
import type { Path } from '$app/types';

export type Stage = 'foundation' | 'alignment' | 'efficiency' | 'derivation' | 'synthesis';

export interface Chapter {
	slug: string;
	/** Number shown in the nav ("01"). */
	n: number;
	title: string;
	short: string;
	/** One-line hook shown on the pipeline map. */
	blurb: string;
	stage: Stage;
	/** False until the chapter's branch is merged; the nav shows it as "coming soon". */
	ready: boolean;
}

export const stageLabel: Record<Stage, string> = {
	foundation: 'Foundation',
	alignment: 'Core post-training',
	efficiency: 'Efficient fine-tuning',
	derivation: 'Community derivations',
	synthesis: 'Putting it together'
};

export const chapters: Chapter[] = [
	{
		slug: 'base-vs-instruct',
		n: 1,
		title: 'Base vs Instruct',
		short: 'Base',
		blurb: 'A pretrained model continues documents. It does not answer.',
		stage: 'foundation',
		ready: false
	},
	{
		slug: 'sft',
		n: 2,
		title: 'Chat templates & SFT',
		short: 'SFT',
		blurb: 'Teach the format: imitate demonstrations, learn only on assistant tokens.',
		stage: 'alignment',
		ready: false
	},
	{
		slug: 'lora',
		n: 3,
		title: 'LoRA & QLoRA',
		short: 'LoRA',
		blurb: 'Train a low-rank update ΔW = BA instead of every weight.',
		stage: 'efficiency',
		ready: false
	},
	{
		slug: 'rlhf',
		n: 4,
		title: 'Reward models & RLHF',
		short: 'RLHF',
		blurb: 'Learn a reward from human comparisons, then optimise against it with PPO.',
		stage: 'alignment',
		ready: false
	},
	{
		slug: 'dpo',
		n: 5,
		title: 'DPO & friends',
		short: 'DPO',
		blurb: 'Skip the reward model: optimise preferences directly from pairs.',
		stage: 'alignment',
		ready: false
	},
	{
		slug: 'rlvr',
		n: 6,
		title: 'RLVR & GRPO',
		short: 'RLVR',
		blurb: 'Reward what a checker can verify. Reasoning emerges.',
		stage: 'alignment',
		ready: false
	},
	{
		slug: 'distillation',
		n: 7,
		title: 'Distillation',
		short: 'Distill',
		blurb: 'A big teacher shapes a small student, token by token.',
		stage: 'efficiency',
		ready: false
	},
	{
		slug: 'abliteration',
		n: 8,
		title: 'Abliteration',
		short: 'Abliterate',
		blurb: 'Refusal lives along one direction. Project it out of the weights.',
		stage: 'derivation',
		ready: false
	},
	{
		slug: 'merging',
		n: 9,
		title: 'Model merging',
		short: 'Merge',
		blurb: 'Add, interpolate and prune task vectors without any training.',
		stage: 'derivation',
		ready: false
	},
	{
		slug: 'case-study',
		n: 10,
		title: 'A Hugging Face family tree',
		short: 'Case study',
		blurb: 'Follow one real model family from base checkpoint to community variants.',
		stage: 'synthesis',
		ready: false
	}
];

export function neighbours(slug: string): { prev?: Chapter; next?: Chapter } {
	const i = chapters.findIndex((c) => c.slug === slug);
	return { prev: chapters[i - 1], next: chapters[i + 1] };
}

/** Base-path-aware link to a chapter (each chapter is its own route folder: src/routes/<slug>/). */
export function chapterHref(slug: string): string {
	return resolve(`${slug}/` as Path);
}
