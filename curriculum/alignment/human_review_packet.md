# CISCE Grade 3–5 STEM alignment — human review packet

**Status: not verified for `ALIGN_STRICT`.** Every outcome in
`cisce_grade_3_5_stem.yaml` is still `needs_human_review` with an empty
`reviewer` field. Do not mark `verified` until a named human fills `reviewer`
and `reviewed_at`.

A human compared the previous 21-row draft to the 2025
[CISCE Primary Curriculum PDF](https://cisce.org/wp-content/uploads/2025/03/PrimaryCurriculum.pdf)
(local copy: `curriculum/raw/_alignment_only/cisce/PrimaryCurriculum.pdf`).
That review is applied in the YAML as a **corrected draft**, not an approval.

The CISCE PDF is alignment-only. It must never be ingested, embedded, or
retrieved. Summaries in the YAML are paraphrases, not excerpts.

## What changed after the PDF review

- Printed Science pages: Class III from **p.154**, Class IV from **p.165**,
  Class V from **p.178**. The old `p.152` citations were the Science section
  opening, not the Class III outcomes.
- Printed Mathematics pages: Class III from **p.82**, Class IV from **p.90**,
  Class V from **p.101**.
- **Removed as CISCE mappings:** Class III fractions; Class III
  weather/seasons/instruments; Class IV Earth/sky/planets; Class IV angle
  module (EngageNY G4 M4); Class IV decimals (EngageNY G4 M6); Class V
  coordinate plane (EngageNY G5 M6).
- **Rewritten or split** remaining rows to official theme names, with
  section allowlists and exclude lists instead of words such as `plant`,
  `animal`, `force`, or `earth`.
- **Gap rows** (`mapped_units: []`) record CISCE themes that still have no
  strict ingested source: Class III Patterns; Class IV Playing with Numbers,
  Geometry, Data Handling, Patterns; Class V large-number operations, factors
  and multiples, negative numbers, geometry, percentage, data, patterns; plus
  some Science hygiene/body rows.

## Still blocking approval

1. No outcome is `verified`.
2. Several EngageNY modules only partly cover a CISCE theme (for example
   Class III number operations is mostly multiplication/division).
3. Section allowlists still need a pass against real ingested headings after
   the next ingest.
4. Utah remains support-only for Grades 4–5 even when a section string matches.

Until a reviewer signs the YAML, exploratory retrieval may still attach
`needs_human_review` IDs. Strict CISCE-aligned retrieval must keep excluding
unmapped chunks (`alignment_strict`).
