import type { RunTrace } from "../types";
import { GenerationOutput } from "./GenerationOutput";

interface Props {
  trace: RunTrace | null;
  running: boolean;
}

export function LiveResponse({ trace, running }: Props) {
  return (
    <section className="live-response" aria-live="polite">
      <header className="panel-header">
        <h2>Generator output</h2>
      </header>
      {trace || running ? (
        <>
          <GenerationOutput trace={trace} running={running} />
        </>
      ) : (
        <p className="muted">Submit a question to stream the tutor response here.</p>
      )}
    </section>
  );
}
