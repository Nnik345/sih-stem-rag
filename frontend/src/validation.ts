import type { QueryFormState } from "./types";

export function validateQueryForm(form: QueryFormState): Record<string, string> {
  const errors: Record<string, string> = {};
  if (!form.query.trim()) {
    errors.query = "Enter a student question.";
  }
  const gradeNum = Number(form.grade);
  if (form.grade && (!Number.isInteger(gradeNum) || gradeNum < 1 || gradeNum > 12)) {
    errors.grade = "Grade must be 1 through 12.";
  }
  if (form.subject && !["science", "mathematics"].includes(form.subject)) {
    errors.subject = "Subject must be science or mathematics.";
  }
  return errors;
}

export function formToPayload(form: QueryFormState): Record<string, unknown> {
  return {
    query: form.query.trim(),
    grade: form.grade ? Number(form.grade) : null,
    subject: form.subject || null,
    unit: form.unit.trim() || null,
    resource_type: form.resource_type.trim() || null,
    audience: form.audience.trim() || null,
    document_id: form.document_id.trim() || null,
    tutor_state: form.tutor_state || null,
    retrieval_only: form.retrieval_only,
    strict: form.strict,
  };
}
