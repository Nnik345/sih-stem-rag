import { useEffect, useRef } from "react";
import type { RunTrace } from "../types";

interface Props {
  trace: RunTrace | null;
  running?: boolean;
}

export function GenerationOutput({ trace, running = false }: Props) {
  const streamRef = useRef<HTMLPreElement>(null);
  const prompt = trace?.prompt;
  const skipped = Boolean(prompt?.generation_skipped || trace?.stages.generator?.status === "skipped");
  const text = trace?.generation?.response_text ?? "";

  useEffect(() => {
    const el = streamRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [text]);

  if (skipped) {
    return (
      <p className="skip" data-testid="generation-skipped">
        {prompt?.skip_reason || trace?.stages.generator?.summary || "Generation was skipped."}
      </p>
    );
  }
  return (
    <pre className="stream" data-testid="generation-stream" ref={streamRef}>
      {text || (running ? "(waiting for tokens)" : "(no response yet)")}
    </pre>
  );
}
