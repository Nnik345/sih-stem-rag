import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { STAGE_ORDER, type RunTrace } from "../types";

export function TimingPanel({ trace }: { trace: RunTrace | null }) {
  const data = STAGE_ORDER.map((name) => ({
    name,
    ms: trace?.stages[name]?.elapsed_ms ?? 0,
  })).filter((row) => row.ms);
  const total = data.reduce((sum, row) => sum + row.ms, 0);
  return (
    <section>
      <h2>Stage timing</h2>
      <p>Total recorded stage time: {total.toFixed(1)} ms</p>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} layout="vertical">
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={80} />
            <Tooltip />
            <Bar dataKey="ms" fill="#2dd4bf" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
