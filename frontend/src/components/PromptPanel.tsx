import type { RunTrace } from "../types";

export function PromptPanel({ trace }: { trace: RunTrace }) {
  const prompt = trace.prompt;
  if (!prompt) return <p>Prompt has not been built.</p>;
  return (
    <section>
      <h2>Socratic prompt and generator</h2>
      <p>Tutor state: <strong>{prompt.tutor_state}</strong></p>
      <h3>System prompt</h3>
      <pre data-testid="system-prompt">{prompt.system_prompt}</pre>
      <h3>User prompt</h3>
      <pre data-testid="user-prompt">{prompt.user_prompt}</pre>
      <h3>Evidence blocks</h3>
      {prompt.evidence_blocks.map((block) => (
        <article key={block.chunk_id} className="evidence-block">
          <h4>E{block.index} · {block.chunk_id}</h4>
          <pre>{JSON.stringify(block.provenance, null, 2)}</pre>
          <p>{block.text}</p>
        </article>
      ))}
      <h3>Generation settings</h3>
      <pre>{JSON.stringify(prompt.generation_settings, null, 2)}</pre>
    </section>
  );
}
