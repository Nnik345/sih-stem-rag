import { useCallback, useEffect, useState } from "react";
import { GraphPanel } from "./components/GraphPanel";
import { LiveResponse } from "./components/LiveResponse";
import { NodeDrawer } from "./components/NodeDrawer";
import { PipelineRail } from "./components/PipelineRail";
import { QueryPanel } from "./components/QueryPanel";
import { AuxPanels, StageDetail } from "./components/StageDetail";
import { applyTraceStages, isTerminalEvent } from "./sse";
import type { GraphNode, QueryFormState, RunTrace, StageName } from "./types";
import { EMPTY_FORM } from "./types";
import { formToPayload, validateQueryForm } from "./validation";

export default function App() {
  const [form, setForm] = useState<QueryFormState>(EMPTY_FORM);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [trace, setTrace] = useState<RunTrace | null>(null);
  const [running, setRunning] = useState(false);
  const [active, setActive] = useState<StageName>("query");
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ api: "unavailable" }));
  }, []);

  const stages = applyTraceStages(trace);

  const startRun = useCallback(async () => {
    const nextErrors = validateQueryForm(form);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length) return;
    setRunning(true);
    setSelected(null);
    setTrace(null);
    try {
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formToPayload(form)),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setErrors({ query: String(body.detail ?? "Could not start run.") });
        setRunning(false);
        return;
      }
      const { run_id } = (await response.json()) as { run_id: string };
      const source = new EventSource(`/api/runs/${run_id}/events`);
      source.onmessage = (message) => {
        try {
          const envelope = JSON.parse(message.data) as { event?: string; trace?: RunTrace };
          if (envelope.trace) setTrace(envelope.trace);
        } catch {
          /* ignore malformed keepalive payloads */
        }
      };
      const named = [
        "run_started",
        "filters_applied",
        "dense_started",
        "dense_completed",
        "lexical_started",
        "lexical_completed",
        "graph_started",
        "graph_completed",
        "fusion_completed",
        "reranker_started",
        "reranker_completed",
        "evidence_completed",
        "prompt_built",
        "generation_started",
        "generation_token",
        "generation_completed",
        "run_completed",
        "run_failed",
      ];
      for (const eventName of named) {
        source.addEventListener(eventName, (message) => {
          const envelope = JSON.parse((message as MessageEvent).data) as {
            event: string;
            trace?: RunTrace;
          };
          if (envelope.trace) setTrace(envelope.trace);
          if (isTerminalEvent(eventName)) {
            source.close();
            setRunning(false);
          }
        });
      }
      source.onerror = () => {
        source.close();
        setRunning(false);
      };
    } catch (error) {
      setErrors({ query: error instanceof Error ? error.message : "Network error" });
      setRunning(false);
    }
  }, [form]);

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>STEM RAG visualizer</h1>
          <p>Local pipeline observer · binds to localhost · no telemetry</p>
        </div>
        <p className="health">
          API {String(health?.api ?? "…")} · Neo4j {String(health?.neo4j ?? "…")}
        </p>
      </header>
      <QueryPanel
        form={form}
        errors={errors}
        running={running}
        onChange={setForm}
        onSubmit={() => void startRun()}
      />
      <LiveResponse trace={trace} running={running} />
      <PipelineRail stages={stages} active={active} onSelect={setActive} />
      <div className="workspace">
        <GraphPanel
          trace={trace}
          selectedId={selected?.node_id ?? null}
          onSelectNode={setSelected}
        />
        <div className="detail-column">
          <StageDetail stage={active} trace={trace} />
          <NodeDrawer node={selected} trace={trace} onClose={() => setSelected(null)} />
        </div>
      </div>
      <AuxPanels trace={trace} />
    </div>
  );
}
