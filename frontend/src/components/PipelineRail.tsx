import { STAGE_ORDER, type StageName, type StageTrace } from "../types";

const LABELS: Record<StageName, string> = {
  query: "Query",
  filters: "Filters",
  dense: "Dense",
  lexical: "Lexical",
  graph: "Graph",
  fusion: "Fusion",
  reranker: "Reranker",
  evidence: "Evidence",
  prompt: "Prompt",
  generator: "Generator",
};

interface Props {
  stages: Record<StageName, StageTrace>;
  active: StageName;
  onSelect: (name: StageName) => void;
}

export function PipelineRail({ stages, active, onSelect }: Props) {
  return (
    <ol className="pipeline-rail">
      {STAGE_ORDER.map((name) => {
        const stage = stages[name];
        const ms = stage.elapsed_ms != null ? `${Math.round(stage.elapsed_ms)} ms` : "";
        return (
          <li key={name}>
            <button
              type="button"
              className={`stage-chip status-${stage.status} ${active === name ? "active" : ""}`}
              onClick={() => onSelect(name)}
            >
              <span className="stage-name">{LABELS[name]}</span>
              <span className="stage-state">{stage.status}</span>
              {ms ? <span className="stage-ms">{ms}</span> : null}
              {stage.summary ? <span className="stage-summary">{stage.summary}</span> : null}
            </button>
          </li>
        );
      })}
    </ol>
  );
}
