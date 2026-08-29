import type { RunTrace } from "../types";

export function EvidencePanel({ trace }: { trace: RunTrace }) {
  const evidence = trace.evidence;
  if (!evidence) return <p>Evidence gate has not completed.</p>;
  return (
    <section>
      <h2>Evidence gate</h2>
      <p className={`verdict ${evidence.sufficient ? "ok" : "bad"}`} data-testid="evidence-verdict">
        {evidence.sufficient ? "SUFFICIENT" : "INSUFFICIENT"} · {evidence.confidence}
      </p>
      <div className="check-grid">
        {evidence.checks.map((check) => (
          <article
            key={check.name}
            className={`check-card ${check.passed ? "pass" : "fail"}`}
            data-testid="evidence-check"
          >
            <h3>{check.name}</h3>
            <p>{check.passed ? "PASS" : "FAIL"}</p>
            <p>Actual: {String(check.value ?? "—")}</p>
            <p>Threshold: {String(check.threshold ?? "—")}</p>
            <p>{check.detail}</p>
          </article>
        ))}
      </div>
      {evidence.reasons.length ? (
        <ul>
          {evidence.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      ) : null}
      <p>Kept chunks: {evidence.kept_chunk_ids.join(", ") || "(none)"}</p>
    </section>
  );
}
