# Reranker evidence-gate calibration

`bge-reranker-v2-m3` emits an **unbounded logit**. That raw value is stored as
both `rerank_score` and `raw_rerank_score`. `rerank_probability` is
`sigmoid(logit)` and is **not** the evidence floor.

| Raw logit | sigmoid | Meaning |
| --------- | ------: | ------- |
| well below 0 | ≪ 0.5 | model leans irrelevant |
| **0.0** | **0.5** | not a “no evidence” cut; chance-level relevance |
| well above 0 | ≫ 0.5 | model leans relevant |

The production floor remains **`min_rerank_score = 0.0` (logit)**. That is a
reviewed starting point: false rejects happen when a true passage scores a
negative logit; false accepts happen when boilerplate or a loosely related
passage scores above 0. Lowering the floor only to pass one weather query is
not allowed. Boilerplate, credits, practice items and answer keys are rejected
**regardless of score**.

## Reviewed pairs (Grade 3–5)

These pairs are for threshold discussion. They are not a claim that CISCE
alignment is verified.

| ID | Query | Chunk (summary) | Label | Expected logit side |
| -- | ----- | --------------- | ----- | ------------------- |
| r1 | how does weather change from day to day | Weather can change from day to day; temperature, wind and clouds | relevant G3 science | ≥ 0 after correct extraction |
| r2 | what instruments measure weather | thermometer, rain gauge, wind vane | relevant G3 science | ≥ 0 |
| r3 | how do black holes evaporate | any Grade 3 weather or plant paragraph | irrelevant | < 0 |
| r4 | how does weather change from day to day | Creative Commons licence footer | irrelevant boilerplate | reject even if logit > 0 |
| r5 | what is a unit fraction | unit fractions on a number line (EngageNY G3 M5) | relevant G3 math | ≥ 0 |
| r6 | homework answer for exit ticket | completed sample response / answer key | unsafe | reject by partition |

Trade-off at logit 0.0: prefer missing a weakly phrased match over answering
from credits, homework solutions or off-grade text. Revisit only with a labelled
set large enough to estimate false-accept and false-reject rates.
