import { useState, type FormEvent } from "react";
import type { QueryFormState } from "../types";
import { validateQueryForm } from "../validation";

interface Props {
  form: QueryFormState;
  errors: Record<string, string>;
  running: boolean;
  onChange: (form: QueryFormState) => void;
  onSubmit: () => void;
}

export function QueryPanel({ form, errors, running, onChange, onSubmit }: Props) {
  const [localErrors, setLocalErrors] = useState<Record<string, string>>({});
  const shown = { ...errors, ...localErrors };

  function update<K extends keyof QueryFormState>(key: K, value: QueryFormState[K]) {
    onChange({ ...form, [key]: value });
    if (localErrors[key as string]) {
      const next = { ...localErrors };
      delete next[key as string];
      setLocalErrors(next);
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const nextErrors = validateQueryForm(form);
    setLocalErrors(nextErrors);
    if (Object.keys(nextErrors).length) return;
    onSubmit();
  }

  return (
    <form className="query-panel" onSubmit={handleSubmit} noValidate>
      <label className="query-label">
        Student question
        <textarea
          value={form.query}
          onChange={(e) => update("query", e.target.value)}
          rows={3}
          placeholder="how does weather change from day to day"
            aria-invalid={Boolean(shown.query)}
        />
        {shown.query ? <span className="field-error">{shown.query}</span> : null}
      </label>
      <div className="query-grid">
        <label>
          Grade
          <select value={form.grade} onChange={(e) => update("grade", e.target.value)}>
            <option value="">Any</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
          </select>
          {shown.grade ? <span className="field-error">{shown.grade}</span> : null}
        </label>
        <label>
          Subject
          <select value={form.subject} onChange={(e) => update("subject", e.target.value)}>
            <option value="">Any</option>
            <option value="science">science</option>
            <option value="mathematics">mathematics</option>
          </select>
          {shown.subject ? <span className="field-error">{shown.subject}</span> : null}
        </label>
        <label>
          Unit ID
          <input value={form.unit} onChange={(e) => update("unit", e.target.value)} />
        </label>
        <label>
          Resource type
          <input value={form.resource_type} onChange={(e) => update("resource_type", e.target.value)} />
        </label>
        <label>
          Audience
          <input value={form.audience} onChange={(e) => update("audience", e.target.value)} />
        </label>
        <label>
          Document ID
          <input value={form.document_id} onChange={(e) => update("document_id", e.target.value)} />
        </label>
        <label>
          Tutor state
          <select value={form.tutor_state} onChange={(e) => update("tutor_state", e.target.value)}>
            <option value="">ASK_QUESTION (default)</option>
            <option value="ASK_QUESTION">ASK_QUESTION</option>
            <option value="GIVE_HINT">GIVE_HINT</option>
            <option value="CORRECT_MISCONCEPTION">CORRECT_MISCONCEPTION</option>
            <option value="EXPLAIN_CONCEPT">EXPLAIN_CONCEPT</option>
            <option value="CONFIRM_STEP">CONFIRM_STEP</option>
          </select>
        </label>
      </div>
      <div className="query-flags">
        <label className="check">
          <input
            type="checkbox"
            checked={form.retrieval_only}
            onChange={(e) => update("retrieval_only", e.target.checked)}
          />
          Retrieval only
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={form.strict}
            onChange={(e) => update("strict", e.target.checked)}
          />
          Strict insufficient-evidence
        </label>
        <button type="submit" disabled={running}>
          {running ? "Running…" : "Run pipeline"}
        </button>
      </div>
    </form>
  );
}
