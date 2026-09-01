# Retrieval evaluation set

This directory holds **manually curated** retrieval questions used to measure the
hybrid GraphRAG retriever. There are deliberately no labels shipped with the
repository: inventing relevance judgements to make the metric script run would
produce meaningless numbers.

## Files

| File | Purpose |
| --- | --- |
| `retrieval_questions.jsonl` | The real evaluation set. Create it yourself; it is not committed. |
| `retrieval_questions.example.jsonl` | Two illustrative records showing the schema. Not a benchmark. |
| `reranker_calibration.md` | Raw logit vs sigmoid, selected evidence floor, and reviewed relevant/irrelevant pairs. |

## Record schema (one JSON object per line)

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `question_id` | string | yes | Stable identifier, unique within the file. |
| `question` | string | yes | The student question, written as a student would ask it. |
| `grade` | integer | yes | Curriculum class to filter on (`1`–`12`). Required. |
| `subject` | string | yes | `"mathematics"`, `"science"`, `"physics"`, `"chemistry"`, or `"biology"`, matching what that class offers. |
| `expected_unit` | string or null | no | `unit_id` (or a distinctive substring of the unit title) where the answer should live. Enables unit-level scoring without chunk labels. |
| `relevant_chunk_ids` | array of strings | no | Chunk IDs judged relevant, from `scripts/test_retriever.py` output. Required for Recall@K / MRR / nDCG. |
| `relevance_grades` | object | no | Optional graded judgements, `{"<chunk_id>": 2}`. Missing chunks in `relevant_chunk_ids` default to grade 1. Only used by nDCG. |
| `expected_insufficient` | boolean | no | `true` if the evidence gate *should* refuse this question (e.g. out-of-curriculum). Defaults to `false`. |
| `notes` | string | no | Free-form annotation: why these chunks, ambiguities, known parsing issues. |

Unknown fields are ignored, so the schema can be extended without breaking the
loader.

## How to label

1. Run the question through the retriever and copy the candidate chunk IDs:

   ```bash
   python scripts/test_retriever.py --query "what is a unit fraction" \
       --grade 3 --subject mathematics --json > /tmp/candidates.json
   ```

2. Read the retrieved text and decide which chunks genuinely answer the
   question. Record only chunks you have actually read.
3. Add the record to `data/evaluation/retrieval_questions.jsonl`.
4. Re-run scoring:

   ```bash
   python scripts/evaluate_retrieval.py --k 5
   ```

Chunk IDs are deterministic (derived from document hash, section and chunk
index), so labels stay valid across re-ingestions **unless** parsing or chunking
parameters change. If you change `chunk_target_tokens`, section merging, or the
parser heuristics, treat existing chunk-level labels as stale and re-verify
them. Unit-level labels (`expected_unit`) survive re-chunking.

## Metrics

`scripts/evaluate_retrieval.py` reports, per stage (fused and reranked) and only
where the necessary labels exist:

- **Recall@K** — fraction of labelled relevant chunks appearing in the top K.
- **MRR** — mean reciprocal rank of the first relevant chunk.
- **nDCG@K** — graded gain, using `relevance_grades` when provided.
- **Unit hit rate@K** — fraction of questions whose `expected_unit` appears in
  the top K. Useful before chunk-level labelling exists.

Questions lacking the labels a metric needs are skipped for that metric and
counted in the report, rather than being scored as zero.
