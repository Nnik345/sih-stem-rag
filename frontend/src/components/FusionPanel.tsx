import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RunTrace } from "../types";

export function FusionPanel({ trace }: { trace: RunTrace }) {
  const fusion = trace.fusion;
  if (!fusion) return <p>Fusion has not completed.</p>;
  const chart = fusion.candidates.map((row) => ({
    name: `#${row.fused_rank}`,
    dense: row.dense_contribution,
    lexical: row.lexical_contribution,
    graph: row.graph_contribution,
  }));
  return (
    <section>
      <h2>RRF fusion</h2>
      <p data-testid="rrf-formula">
        {fusion.formula} with k={fusion.rrf_k}, weights dense={fusion.weight_dense}, lexical=
        {fusion.weight_fulltext}, graph={fusion.weight_graph}
      </p>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chart}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="dense" stackId="a" fill="#2563eb" />
            <Bar dataKey="lexical" stackId="a" fill="#0e7c7b" />
            <Bar dataKey="graph" stackId="a" fill="#7c3aed" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table>
        <thead>
          <tr>
            <th>Fused rank</th>
            <th>Chunk</th>
            <th>Dense</th>
            <th>Lexical</th>
            <th>Graph</th>
            <th>RRF</th>
            <th>Channels</th>
          </tr>
        </thead>
        <tbody>
          {fusion.candidates.map((row) => (
            <tr key={row.chunk_id} data-testid="fusion-row">
              <td>{row.fused_rank}</td>
              <td>{row.text.slice(0, 80)}</td>
              <td>{row.dense_contribution.toFixed(5)} (r{row.dense_rank ?? "-"})</td>
              <td>{row.lexical_contribution.toFixed(5)} (r{row.lexical_rank ?? "-"})</td>
              <td>{row.graph_contribution.toFixed(5)} (r{row.graph_rank ?? "-"})</td>
              <td>{row.rrf_score.toFixed(5)}</td>
              <td>{row.channels.join("+")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
