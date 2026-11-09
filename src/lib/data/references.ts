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
	},
	{
		id: 'hf-chat-templates',
		authors: 'Hugging Face',
		year: 2025,
		title: 'Chat templates (Transformers documentation)',
		url: 'https://huggingface.co/docs/transformers/main/en/chat_templating',
		cite: 'HF docs, Chat templates'
	},
	{
		id: 'urial',
		authors: 'Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, et al.',
		year: 2023,
		title: 'The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning',
		url: 'https://arxiv.org/abs/2312.01552',
		cite: 'Lin et al., 2023'
	},
	{
		id: 'llama3',
		authors: 'Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al.',
		year: 2024,
		title: 'The Llama 3 Herd of Models',
		url: 'https://arxiv.org/abs/2407.21783',
		cite: 'Grattafiori et al., 2024'
	},
	{
		id: 'olmo2',
		authors: 'Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, et al.',
		year: 2024,
		title: '2 OLMo 2 Furious',
		url: 'https://arxiv.org/abs/2501.00656',
		cite: 'Team OLMo, 2024'
	},
	{
		id: 'olmo3',
		authors: 'Team Olmo',
		year: 2025,
		title: 'Olmo 3',
		url: 'https://arxiv.org/abs/2512.13961',
		cite: 'Team Olmo, 2025'
	},
	{
		id: 'qwen3',
		authors: 'An Yang, Anfeng Li, Baosong Yang, et al.',
		year: 2025,
		title: 'Qwen3 Technical Report',
		url: 'https://arxiv.org/abs/2505.09388',
		cite: 'Yang et al., 2025'
	},
	{
		id: 'qwen3-8b-card',
		authors: 'Qwen Team',
		year: 2025,
		title: 'Qwen3-8B model card',
		url: 'https://huggingface.co/Qwen/Qwen3-8B',
		cite: 'Qwen3-8B card'
	},
	{
		id: 'smollm3',
		authors: 'Elie Bakouch, et al. (Hugging Face)',
		year: 2025,
		title: 'SmolLM3: smol, multilingual, long-context reasoner',
		url: 'https://huggingface.co/blog/smollm3',
		cite: 'Bakouch et al., 2025'
	},
	{
		id: 'nucleus',
		authors: 'Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi',
		year: 2019,
		title: 'The Curious Case of Neural Text Degeneration',
		url: 'https://arxiv.org/abs/1904.09751',
		venue: 'ICLR 2020',
		cite: 'Holtzman et al., 2019'
	},
	{
		id: 'minp',
		authors: 'Minh Nhat Nguyen, Andrew Baker, Clement Neo, Allen Roush, et al.',
		year: 2024,
		title: 'Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs',
		url: 'https://arxiv.org/abs/2407.01082',
		venue: 'ICLR 2025',
		cite: 'Nguyen et al., 2024'
	},
	{
		id: 'lima',
		authors: 'Chunting Zhou, Pengfei Liu, Puxin Xu, et al.',
		year: 2023,
		title: 'LIMA: Less Is More for Alignment',
		url: 'https://arxiv.org/abs/2305.11206',
		venue: 'NeurIPS 2023',
		cite: 'Zhou et al., 2023'
	},
	{
		id: 'tulu3',
		authors: 'Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, et al.',
		year: 2024,
		title: 'Tulu 3: Pushing Frontiers in Open Language Model Post-Training',
		url: 'https://arxiv.org/abs/2411.15124',
		cite: 'Lambert et al., 2024'
	},
	{
		id: 'flan',
		authors: 'Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, et al.',
		year: 2021,
		title: 'Finetuned Language Models Are Zero-Shot Learners',
		url: 'https://arxiv.org/abs/2109.01652',
		venue: 'ICLR 2022',
		cite: 'Wei et al., 2021'
	},
	{
		id: 'trl-sft',
		authors: 'Hugging Face',
		year: 2025,
		title: 'SFT Trainer (TRL documentation)',
		url: 'https://huggingface.co/docs/trl/sft_trainer',
		cite: 'TRL docs, SFT Trainer'
	},
	{
		id: 'smollm2',
		authors: 'Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, et al.',
		year: 2025,
		title: 'SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model',
		url: 'https://arxiv.org/abs/2502.02737',
		cite: 'Ben Allal et al., 2025'
	}
];

export const referenceById = new Map(references.map((r) => [r.id, r]));
