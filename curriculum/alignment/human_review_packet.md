# CISCE Grade 3–5 STEM alignment — human review packet

**Status: not verified.** Every outcome below is `needs_human_review` with an
empty `reviewer` field. Do not mark `verified` until a named human reviewer
fills `reviewer` and `reviewed_at`. Machine-generated paraphrases and unit
mappings are a draft crosswalk only.

The CISCE PDF is alignment-only. It must never be ingested, embedded, or
retrieved. Short supporting excerpts below are for page-checking; confirm
against the local file `curriculum/raw/_alignment_only/cisce/PrimaryCurriculum.pdf`.

Printed Class III Mathematics begins near printed p.62 (PDF p.93 in the 2025
file). Class III Science begins near printed p.152 (PDF p.152 cited in the
YAML). Class IV/V page numbers in the YAML still need confirmation against that
PDF.

For each outcome the reviewer must resolve:

1. Does the paraphrase match the official wording’s intent without overclaiming?
2. Is the cited CISCE page correct for that grade and subject?
3. Do the mapped EngageNY / Siyavula / Utah units actually teach that outcome?
4. For Science, are `section_patterns` too broad or too narrow?
5. Should Utah remain support-only for Grades 4–5 even when a section matches?

| Local ID | Grade | Subject | Paraphrase | CISCE page (draft) | Mapped source units | Excerpt / questions |
| -------- | ----: | ------- | ---------- | ------------------ | ------------------- | ------------------- |
| cisce_g3_math_number_operations | 3 | mathematics | Use place value and the four operations to solve whole-number problems. | PrimaryCurriculum.pdf (Mathematics, Class III) | EngageNY G3 M1, M3 | Confirm four-operations wording vs PDF p.93 region. |
| cisce_g3_math_measurement | 3 | mathematics | Measure and compare length, mass, capacity and time with standard units. | same | G3 M2 | Confirm mass/capacity vs EngageNY measure module. |
| cisce_g3_math_fractions | 3 | mathematics | Recognise unit fractions and locate them on a number line. | same | G3 M5 | Confirm “unit fraction” language in CISCE. |
| cisce_g3_math_geometry | 3 | mathematics | Describe plane shapes, area of rectangles, and simple perimeter problems. | same | G3 M4, M7 | Split area vs geometry if CISCE separates them. |
| cisce_g3_math_data | 3 | mathematics | Collect, display and read simple categorical data. | same | G3 M6 | Confirm data handling is Class III. |
| cisce_g4_math_place_value | 4 | mathematics | Use place value to round and compute with multi-digit whole numbers. | Class IV | G4 M1, M3 | Confirm printed Class IV page. |
| cisce_g4_math_measurement | 4 | mathematics | Convert metric units and solve measurement problems. | Class IV | G4 M2, M7 | Metric-only vs imperial mentions. |
| cisce_g4_math_fractions_decimals | 4 | mathematics | Compare fractions and relate tenths and hundredths to decimal notation. | Class IV | G4 M5, M6 | Two modules vs one CISCE strand. |
| cisce_g4_math_geometry | 4 | mathematics | Measure angles and classify plane figures. | Class IV | G4 M4 | Confirm angle measure in Class IV. |
| cisce_g5_math_decimals | 5 | mathematics | Use place value with decimals and operate on decimal numbers. | Class V | G5 M1, M2 | Confirm Class V page. |
| cisce_g5_math_fractions | 5 | mathematics | Add, subtract, multiply and divide fractions in problem contexts. | Class V | G5 M3, M4 | Division of fractions in CISCE? |
| cisce_g5_math_volume_coordinates | 5 | mathematics | Relate volume to multiplication and locate points in the first quadrant. | Class V | G5 M5, M6 | Volume and coordinates may be separate CISCE bullets. |
| cisce_g3_sci_living_world | 3 | science | Observe organisms, habitats and how living things use resources. | PDF p.152 Class III | Utah G3 sections matching organism/habitat/plant/animal | Patterns may over-map generic “plant” pages. |
| cisce_g3_sci_earth_weather | 3 | science | Describe weather, seasons and simple Earth processes from observation. | PDF p.152 Class III | Utah G3 weather/season/instrument sections | Confirm weather vs climate wording. |
| cisce_g3_sci_forces_materials | 3 | science | Explore motion, forces and properties of everyday materials. | PDF p.152 Class III | Utah G3 force/magnet/material sections | Materials vs forces: one outcome or two in CISCE? |
| cisce_g4_sci_life | 4 | science | Describe plants, animals and life processes using primary learner text. | Class IV (confirm page) | Siyavula G4 + Utah G4 support | Prefer Siyavula; Utah is support. |
| cisce_g4_sci_matter_energy | 4 | science | Investigate matter, materials, energy and change in everyday contexts. | Class IV (confirm page) | Siyavula G4 + Utah G4 | “Energy” pattern may hit many chapters. |
| cisce_g4_sci_earth | 4 | science | Relate Earth features, soil, water and sky observations to local evidence. | Class IV (confirm page) | Siyavula G4 + Utah G4 | Sky vs weather overlap with G3. |
| cisce_g5_sci_life | 5 | science | Explain life processes and ecosystems with Grade 5 learner evidence. | Class V (confirm page) | Siyavula G5 + Utah G5 | Ecosystem vs Class V life science. |
| cisce_g5_sci_matter_energy | 5 | science | Use particle ideas, energy transfers and simple investigations of materials. | Class V (confirm page) | Siyavula G5 + Utah G5 | Particle ideas may exceed CISCE primary. |
| cisce_g5_sci_earth_space | 5 | science | Connect Earth systems and the solar system to Grade 5 learner text. | Class V (confirm page) | Siyavula G5 + Utah G5 | Solar system presence in Siyavula G5? |

Until this review is signed, retrieval may use `needs_human_review` mappings in
exploratory tutor mode. Strict CISCE-aligned retrieval must exclude unmapped
chunks (`alignment_strict`).
