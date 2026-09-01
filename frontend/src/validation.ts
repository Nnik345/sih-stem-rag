import type { QueryFormState } from "./types";
import { subjectsForGrade } from "./curriculum";

export function validateQueryForm(form: QueryFormState): Record<string, string> {
  const errors: Record<string, string> = {};
  if (!form.query.trim() && !form.imageFile) {
    errors.query = "Enter a student question or attach an image.";
  }
  const gradeNum = Number(form.grade);
  if (!form.grade) {
    errors.grade = "Grade is required.";
  } else if (!Number.isInteger(gradeNum) || gradeNum < 1 || gradeNum > 12) {
    errors.grade = "Grade must be 1 through 12.";
  }
  if (!form.subject) {
    errors.subject = "Subject is required.";
  } else if (form.grade && !subjectsForGrade(gradeNum).includes(form.subject)) {
    errors.subject = `Subject is not offered in class ${form.grade}.`;
  }
  return errors;
}

export function formToPayload(form: QueryFormState): Record<string, unknown> {
  return {
    query: form.query.trim(),
    grade: Number(form.grade),
    subject: form.subject,
    tutor_state: form.tutor_state || null,
    retrieval_only: form.retrieval_only,
    strict: form.strict,
  };
}
