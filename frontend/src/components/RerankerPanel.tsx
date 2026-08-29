import { useMemo, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RunTrace } from "../types";

export function RerankerPanel({ trace }: { trace: RunTrace }) {
  const reranker = trace.reranker;
  const [sortKey, setSortKey] = useState<"reranked_rank" | "movement">("reranked_rank");
  const rows = useMemo(() => {
    const list = [...(reranker?.candidates ?? [])];
    list.sort((a, b) =>
      sortKey === "movement"
        ? b.rank_movement - a.rank_movement
        : (a.reranked_rank ?? 0) - (b.reranked_rank ?? 0),
    );
    return list;
  }, [reranker, sortKey]);
  if (!reranker) return <p>Reranking has not completed (or was skipped).</p>;
  return (
    <section>
      <h2>Reranker</h2>
      <p className="muted">
        Scores are raw BGE relevance logits, comparable within this query only. They do not explain
        the model&apos;s internal reasoning.
      </p>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={rows.map((row) => ({ name: row.chunk_id.slice(-8), movement: row.rank_movement }))}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="movement" fill="#a855f7" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table>
        <thead>
          <tr>
            <th>Fused rank</th>
            <th><button type="button" onClick={() => setSortKey("reranked_rank")}>Reranked rank</button></th>
            <th><button type="button" onClick={() => setSortKey("movement")}>Movement</button></th>
            <th>Logit</th>
            <th>Preview</th>
            <th>final_top_k</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.chunk_id} data-testid="rerank-row">
              <td>{row.fused_rank}</td>
              <td>{row.reranked_rank}</td>
              <td data-testid="rank-movement">{row.rank_movement > 0 ? `+${row.rank_movement}` : row.rank_movement}</td>
              <td>{row.rerank_score?.toFixed(4)}</td>
              <td>{row.text.slice(0, 90)}</td>
              <td>{row.survived_final_top_k ? "kept" : "removed"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
