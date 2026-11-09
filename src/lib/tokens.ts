/** Display helpers for tokens: make whitespace visible and spot special/control tokens. */

export function isSpecial(token: string): boolean {
	return /^<\|.*\|>$/.test(token) || /^<\/?think>$/.test(token);
}

/** Newlines become ↵ so they're visible in a chip; leading spaces are kept (rendered with pre). */
export function showToken(token: string): string {
	return token.replaceAll('\n', '↵');
}
