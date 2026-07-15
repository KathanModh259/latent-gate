export function sanitizeHtml(input: string): string {
  const div = document.createElement('div');
  div.textContent = input;
  return div.innerHTML;
}

export function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
}

export function validateInput(text: string, maxChars: number = 50000): { valid: boolean; error?: string } {
  if (!text || !text.trim()) {
    return { valid: false, error: 'Input cannot be empty' };
  }
  if (text.length > maxChars) {
    return { valid: false, error: `Input exceeds maximum length of ${maxChars.toLocaleString()} characters` };
  }
  return { valid: true };
}