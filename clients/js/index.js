/**
 * Confab client — drop-in fetch wrapper that adds hallucination confidence.
 *
 * Usage:
 *   import { confab } from 'confab-client';
 *   const client = confab({ baseUrl: 'http://localhost:8080' });
 *   const result = await client.chat('What is Python?');
 *   console.log(result.content);    // raw response
 *   console.log(result.claims);     // [{text, confidence, level}]
 *   console.log(result.confidence); // average 0-1
 */

function confab({ baseUrl = 'http://localhost:8080', model = 'gpt-4o-mini' } = {}) {
  return {
    async chat(prompt, options = {}) {
      const resp = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: options.model || model,
          messages: [{ role: 'user', content: prompt }],
          ...options,
        }),
      });

      if (!resp.ok) throw new Error(`Confab proxy error: ${resp.status}`);
      const data = await resp.json();

      const content = data.choices[0].message.content;
      const claims = data.confab_metadata?.claims || [];
      const confidence = claims.length
        ? claims.reduce((sum, c) => sum + c.confidence, 0) / claims.length
        : 1.0;

      return { content, claims, confidence, raw: data };
    },

    async *chatStream(prompt, options = {}) {
      const resp = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: options.model || model,
          messages: [{ role: 'user', content: prompt }],
          stream: true,
          ...options,
        }),
      });

      if (!resp.ok) throw new Error(`Confab proxy error: ${resp.status}`);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') return;
          const chunk = JSON.parse(payload);
          yield chunk;
        }
      }
    },
  };
}

module.exports = { confab };
