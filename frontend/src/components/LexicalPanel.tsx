import { useMemo, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RunTrace } from "../types";

export function LexicalPanel({ trace }: { trace: RunTrace }) {
  const lexical = trace.lexical;
  const [sortKey, setSortKey] = useState<"rank" | "score">("rank");
  const rows = useMemo(() => {
    const list = [...(lexical?.candidates ?? [])];
    list.sort((a, b) =>
      sortKey === "score" ? (b.score ?? 0) - (a.score ?? 0) : (a.rank ?? 0) - (b.rank ?? 0),
    );
    return list;
  }, [lexical, sortKey]);
  if (!lexical) return <p>Lexical retrieval has not completed.</p>;
  return (
    <section>
      <h2>Lexical retrieval</h2>
      <p>Original query</p>
      <pre>{lexical.original_query}</pre>
      <p>Generated Lucene query</p>
      <pre data-testid="lucene-query">{lexical.lucene_query}</pre>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={rows.map((row) => ({ name: `#${row.rank}`, score: row.score }))}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="score" fill="#0e7c7b" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table>
        <thead>
          <tr>
            <th><button type="button" onClick={() => setSortKey("rank")}>Rank</button></th>
            <th><button type="button" onClick={() => setSortKey("score")}>Full-text score</button></th>
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
              <td>{String(row.provenance.unit_title ?? "")}</td>
              <td>{row.final_evidence ? "evidence" : row.entered_fusion ? "fused" : "dropped"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
