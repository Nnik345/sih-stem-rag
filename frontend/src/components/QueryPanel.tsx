import { useEffect, useState, type FormEvent } from "react";
import { GRADES, subjectsForGrade } from "../curriculum";
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
  const [subjectsByGrade, setSubjectsByGrade] = useState<Record<string, string[]> | null>(null);
  const shown = { ...errors, ...localErrors };
  const gradeNum = Number(form.grade);
  const subjects =
    (form.grade && subjectsByGrade?.[form.grade]) ||
    (Number.isInteger(gradeNum) ? subjectsForGrade(gradeNum) : []);

  useEffect(() => {
    if (typeof fetch !== "function") return;
    fetch("/api/curriculum-options")
      .then((response) => (response.ok ? response.json() : null))
      .then((data: { subjects_by_grade?: Record<string, string[]> } | null) => {
        if (data?.subjects_by_grade) setSubjectsByGrade(data.subjects_by_grade);
      })
      .catch(() => undefined);
  }, []);

  function update<K extends keyof QueryFormState>(key: K, value: QueryFormState[K]) {
    let next: QueryFormState = { ...form, [key]: value };
    if (key === "grade") {
      const allowed = subjectsForGrade(Number(value));
      if (next.subject && !allowed.includes(next.subject)) {
        next = { ...next, subject: "" };
      }
    }
    onChange(next);
    if (localErrors[key as string]) {
      const cleared = { ...localErrors };
      delete cleared[key as string];
      setLocalErrors(cleared);
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
          placeholder="what are the components of food"
          aria-invalid={Boolean(shown.query)}
        />
        {shown.query ? <span className="field-error">{shown.query}</span> : null}
      </label>
      <label className="query-label">
        Student photo (optional)
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif"
          onChange={(e) => update("imageFile", e.target.files?.[0] ?? null)}
        />
        {form.imageFile ? (
          <span className="muted">{form.imageFile.name}</span>
        ) : (
          <span className="muted">JPEG, PNG, WebP, or GIF. Question may be empty if a photo is attached.</span>
        )}
      </label>
      <div className="query-grid">
        <label>
          Grade
          <select
            value={form.grade}
            required
            onChange={(e) => update("grade", e.target.value)}
          >
            <option value="">Select class</option>
            {GRADES.map((g) => (
              <option key={g} value={String(g)}>
                {g}
              </option>
            ))}
          </select>
          {shown.grade ? <span className="field-error">{shown.grade}</span> : null}
        </label>
        <label>
          Subject
          <select
            value={form.subject}
            required
            disabled={!form.grade}
            onChange={(e) => update("subject", e.target.value)}
          >
            <option value="">{form.grade ? "Select subject" : "Select class first"}</option>
            {subjects.map((subject) => (
              <option key={subject} value={subject}>
                {subject}
              </option>
            ))}
          </select>
          {shown.subject ? <span className="field-error">{shown.subject}</span> : null}
        </label>
        <label>
          Tutor state
          <select value={form.tutor_state} onChange={(e) => update("tutor_state", e.target.value)}>
            <option value="">GIVE_HINT (default)</option>
            <option value="GIVE_HINT">GIVE_HINT</option>
            <option value="EXPLAIN_CONCEPT">EXPLAIN_CONCEPT</option>
            <option value="CONFIRM_ANSWER">CONFIRM_ANSWER</option>
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
