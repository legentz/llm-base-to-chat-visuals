/**
 * Single source of truth for citations. Every <Cite id="…"> must point to an entry here
 * (enforced by references.spec.ts). Each entry was checked against its primary source;
 * keep `url` pointing to the canonical page (arXiv abs, official blog, HF model card).
 */
export interface Reference {
	id: string;
	authors: string;
	year: number;
	title: string;
	url: string;
	venue?: string;
	/** Short label used inline, e.g. "Ouyang et al., 2022". */
	cite: string;
}

export const references: Reference[] = [
	{
		id: 'transformer-explainer',
		authors: 'Aeree Cho, Grace C. Kim, Alexander Karpekov, et al.',
		year: 2024,
		title: 'Transformer Explainer: Interactive Learning of Text-Generative Models',
		url: 'https://arxiv.org/abs/2408.04619',
		venue: 'IEEE VIS 2024 (poster)',
		cite: 'Cho et al., 2024'
	},
	{
		id: 'instructgpt',
		authors: 'Long Ouyang, Jeff Wu, Xu Jiang, et al.',
		year: 2022,
		title: 'Training language models to follow instructions with human feedback',
		url: 'https://arxiv.org/abs/2203.02155',
		venue: 'NeurIPS 2022',
		cite: 'Ouyang et al., 2022'
	}
];

export const referenceById = new Map(references.map((r) => [r.id, r]));
