import { useMemo, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RunTrace } from "../types";

export function DensePanel({ trace }: { trace: RunTrace }) {
  const dense = trace.dense;
  const [sortKey, setSortKey] = useState<"rank" | "score">("rank");
  const rows = useMemo(() => {
    const list = [...(dense?.candidates ?? [])];
    list.sort((a, b) => {
      if (sortKey === "score") return (b.score ?? 0) - (a.score ?? 0);
      return (a.rank ?? 0) - (b.rank ?? 0);
    });
    return list;
  }, [dense, sortKey]);
  if (!dense) return <p>Dense retrieval has not completed.</p>;
  return (
    <section>
      <h2>Dense retrieval</h2>
      <p>
        Model <code>{dense.model_name}</code> · dim {dense.embedding_dim} · norm{" "}
        {dense.query_vector_norm?.toFixed(4)} · strategy <strong>{dense.strategy}</strong>
        {dense.used_exact_fallback ? " (exact filtered fallback)" : " (approximate index)"}
      </p>
      <p className="muted">Query vector preview (not the full embedding): {JSON.stringify(dense.vector_preview)}</p>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={rows.map((row) => ({ name: `#${row.rank}`, score: row.score }))}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="score" fill="#2563eb" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table>
        <thead>
          <tr>
            <th><button type="button" onClick={() => setSortKey("rank")}>Rank</button></th>
            <th><button type="button" onClick={() => setSortKey("score")}>Cosine</button></th>
            <th>Chunk</th>
            <th>Provenance</th>
            <th>Later status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.chunk_id}>
              <td>{row.rank}</td>
              <td>{row.score?.toFixed(4)}</td>
              <td>{row.text.slice(0, 120)}</td>
              <td>{String(row.provenance.unit_title ?? "")} p.{String(row.provenance.pages ?? "")}</td>
              <td>
                {row.final_evidence ? "evidence" : row.entered_fusion ? "fused" : "dropped"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
