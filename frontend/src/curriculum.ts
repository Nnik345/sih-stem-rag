/** Mirrors rag.curriculum_catalog.subjects_for_grade — keep in sync with the API. */

export function subjectsForGrade(grade: number): string[] {
  if (grade === 1 || grade === 2) return ["mathematics"];
  if (grade >= 3 && grade <= 10) return ["mathematics", "science"];
  if (grade === 11 || grade === 12) {
    return ["mathematics", "physics", "chemistry", "biology"];
  }
  return [];
}

export const GRADES = Array.from({ length: 12 }, (_, i) => i + 1);
