import type { RunTrace, SseEnvelope, StageName, StageTrace } from "./types";
import { emptyStages } from "./types";

export function applyTraceStages(trace: RunTrace | null): Record<StageName, StageTrace> {
  const stages = emptyStages();
  if (!trace?.stages) return stages;
  for (const [name, stage] of Object.entries(trace.stages)) {
    if (name in stages) {
      stages[name as StageName] = stage;
    }
  }
  return stages;
}

export function parseSseBlock(block: string): { event: string; data: SseEnvelope | null } | null {
  const lines = block.split("\n");
  let event = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return event ? { event, data: null } : null;
  try {
    const parsed = JSON.parse(dataLines.join("\n")) as SseEnvelope;
    return { event: parsed.event || event, data: parsed };
  } catch {
    return { event, data: null };
  }
}

export function isTerminalEvent(event: string): boolean {
  return event === "run_completed" || event === "run_failed";
}
