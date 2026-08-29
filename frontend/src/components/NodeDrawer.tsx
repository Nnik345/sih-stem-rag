import type { GraphNode, GraphPath, RunTrace } from "../types";

interface Props {
  node: GraphNode | null;
  trace: RunTrace | null;
  onClose: () => void;
}

export function NodeDrawer({ node, trace, onClose }: Props) {
  if (!node) return null;
  const paths: GraphPath[] = (trace?.graph?.paths ?? []).filter((path) =>
    node.path_ids.includes(path.path_id),
  );
  return (
    <aside className="node-drawer" data-testid="node-drawer">
      <header>
        <h3>{node.display_label}</h3>
        <button type="button" onClick={onClose} aria-label="Close node details">
          Close
        </button>
      </header>
      <p className="mono">{node.node_id}</p>
      <dl>
        <div><dt>Type</dt><dd>{node.label} · {node.node_kind}</dd></div>
        <div><dt>Status</dt><dd>{node.status}</dd></div>
        {node.seed_weight != null ? (
          <div><dt>Seed weight</dt><dd>{node.seed_weight}</dd></div>
        ) : null}
        {node.graph_score != null ? (
          <div><dt>Graph score</dt><dd>{node.graph_score}</dd></div>
        ) : null}
        <div><dt>Entered fusion</dt><dd>{node.entered_fusion ? "yes" : "no"}</dd></div>
        <div><dt>Final evidence</dt><dd>{node.final_evidence ? "yes" : "no"}</dd></div>
      </dl>
      {node.label === "Chunk" && node.text ? <p className="chunk-text">{node.text}</p> : null}
      <h4>Curriculum metadata</h4>
      <pre>{JSON.stringify({ ...node.metadata, ...{} }, null, 2)}</pre>
      <h4>Paths ({paths.length})</h4>
      <ul className="path-list">
        {paths.map((path) => (
          <li key={path.path_id} data-testid="graph-path">
            <strong>{path.relation}</strong>
            {path.logical ? " (logical)" : ""} from {path.seed_chunk_id}
            {path.via ? ` via ${path.via}` : ""}
            <div className={path.accepted ? "ok" : "bad"}>
              {path.accepted ? "accepted" : "ignored"} · {path.reason_code}
            </div>
            <p>{path.reason_detail}</p>
          </li>
        ))}
      </ul>
    </aside>
  );
}
