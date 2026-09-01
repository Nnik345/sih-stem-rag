import type { RunTrace, StageName } from "../types";
import { DensePanel } from "./DensePanel";
import { LexicalPanel } from "./LexicalPanel";
import { FusionPanel } from "./FusionPanel";
import { RerankerPanel } from "./RerankerPanel";
import { EvidencePanel } from "./EvidencePanel";
import { PromptPanel } from "./PromptPanel";
import { TimingPanel } from "./TimingPanel";

interface Props {
  stage: StageName;
  trace: RunTrace | null;
}

export function StageDetail({ stage, trace }: Props) {
  if (!trace) {
    return <p className="empty">Submit a query to inspect each pipeline stage.</p>;
  }
  switch (stage) {
    case "dense":
      return <DensePanel trace={trace} />;
    case "lexical":
      return <LexicalPanel trace={trace} />;
    case "fusion":
      return <FusionPanel trace={trace} />;
    case "reranker":
      return <RerankerPanel trace={trace} />;
    case "evidence":
      return <EvidencePanel trace={trace} />;
    case "prompt":
    case "generator":
      return <PromptPanel trace={trace} />;
    case "query":
    case "filters":
      return (
        <section>
          <h2>{stage === "query" ? "Query" : "Metadata filters"}</h2>
          <pre>{JSON.stringify(stage === "query" ? { query: trace.query } : trace.filters, null, 2)}</pre>
        </section>
      );
    case "rewrite":
      return (
        <section>
          <h2>Query rewrite</h2>
          <pre>
            {JSON.stringify(
              trace.rewrite ?? {
                original_query: trace.query,
                retrieval_query: null,
              },
              null,
              2,
            )}
          </pre>
        </section>
      );
    case "image":
      return (
        <section>
          <h2>Textbook figure kNN</h2>
          <pre>{JSON.stringify(trace.image ?? { hits: [], skipped: true }, null, 2)}</pre>
        </section>
      );
    case "graph":
      return (
        <section>
          <h2>Graph summary</h2>
          <p>Use the interactive graph on the left. Selected candidate IDs:</p>
          <pre>{JSON.stringify(trace.graph?.selected_chunk_ids ?? [], null, 2)}</pre>
          {(trace.graph?.disabled_relations ?? []).map((item) => (
            <p key={item.relation}>
              {item.relation} disabled: {item.detail}
            </p>
          ))}
        </section>
      );
    default:
      return null;
  }
}

export function AuxPanels({ trace }: { trace: RunTrace | null }) {
  return (
    <div className="aux-panels">
      <TimingPanel trace={trace} />
    </div>
  );
}
