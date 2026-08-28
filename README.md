# SIH STEM RAG

Local research project building a **hallucination-resistant Socratic STEM tutor**
over Core Knowledge Foundation curriculum materials (Grades 1–3 Mathematics and
Science).

The system implements hybrid GraphRAG retrieval on a local Neo4j instance, feeding a
locally hosted Qwen3-VL-8B-Instruct generator that is instructed to tutor rather
than to answer outright.

## Contents

- [First-time local setup](#first-time-local-setup)
- [What is gitignored and how to restore it](#what-is-gitignored-and-how-to-restore-it)
- [Architecture](#architecture)
- [Python version](#python-version)
- [Dataset](#dataset)
- [Models](#models)
- [Neo4j setup](#neo4j-setup)
- [Graph schema](#graph-schema)
- [Retrieval pipeline](#retrieval-pipeline)
- [Evidence sufficiency gate](#evidence-sufficiency-gate)
- [Socratic controller](#socratic-controller)
- [Commands](#commands)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Project structure](#project-structure)
- [Current limitations](#current-limitations)
- [Future experimental directions](#future-experimental-directions)

## First-time local setup

Everything runs locally. Neo4j is installed from tarballs under `.neo4j-local/`
and managed with [`scripts/neo4j_local.sh`](scripts/neo4j_local.sh) — no system
packages, no root, and no Docker. Large downloads and generated data are
git-ignored; see the next section for how to restore them after a fresh clone.

**Prerequisites:** Python **3.12** and ~25 GB free disk for models, corpus, and
Neo4j (more during ingestion).

### 1. Python environment

```bash
python -m venv .venv          # optional; .venv/ is git-ignored
source .venv/bin/activate     # omit if not using a venv
pip install -r requirements.txt
```

### 2. Environment file

```bash
cp .env.example .env
```

Edit `.env` and set `NEO4J_PASSWORD` to a real password. The same value is used by
Neo4j and by every script that connects to the database.

### 3. Curriculum PDFs (~935 MB)

```bash
python scripts/download_core_knowledge_stem.py
```

Output: `core_knowledge_stem/` (git-ignored). Details in [Dataset](#dataset).

### 4. Retrieval models (~4.6 GB total)

```bash
python scripts/download_retrieval_models.py
```

Output: `models/bge-m3/` (~2.3 GB) and `models/bge-reranker-v2-m3/` (~2.3 GB).
If a model is already complete locally, the download script skips it and prints a
status table. Use `--force` to re-download. Details in [Models](#models).

### 5. Generator model (~17 GB)

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir models/qwen3-vl-8b-instruct
```

Output: `models/qwen3-vl-8b-instruct/` (git-ignored). This script is **not** run
by `download_retrieval_models.py`.

### 6. Neo4j + JDK 21

Neo4j 5.x needs **Java 17 or 21**, so a Temurin JDK 21 is installed alongside
Neo4j under `.neo4j-local/` and used only by this project. Your system JDK is
never touched and its version does not matter.

Full steps are in [Neo4j setup](#neo4j-setup). Summary:

```bash
mkdir -p .neo4j-local && cd .neo4j-local
curl -LO https://dist.neo4j.org/neo4j-community-5.26.12-unix.tar.gz
# Download any Temurin JDK 21 Linux x64 tarball from https://adoptium.net/
# and save it here (filename does not matter, e.g. temurin-jdk21.tar.gz)
curl -LO "<your-temurin-21-tarball-url>"
tar --no-same-owner -xzf neo4j-community-5.26.12-unix.tar.gz
tar --no-same-owner -xzf <your-jdk-tarball>
cd ..
./scripts/neo4j_local.sh set-password "<same-as-NEO4J_PASSWORD-in-.env>"
./scripts/neo4j_local.sh start
```

`scripts/neo4j_local.sh` auto-discovers `neo4j-community-*` and `jdk-21*`
directories under `.neo4j-local/`. The `.tar.gz` files (~330 MB combined) are
git-ignored and **safe to delete after extraction** to save disk.

### 7. Initialize schema and ingest the corpus

```bash
python scripts/init_neo4j.py
python scripts/ingest_corpus.py    # ~8 minutes for all 126 PDFs
```

Ingestion writes git-ignored artefacts under `data/processed/` (images,
manifests, cache) and populates Neo4j. Re-running is idempotent unless you pass
`--force`.

### 8. Smoke verification

```bash
python scripts/test_retriever.py \
    --query "why does the moon look different on different nights" \
    --grade 1 --subject science

python scripts/test_rag.py \
    --query "why does the moon look different on different nights" \
    --grade 1 --subject science

python scripts/test_generator.py
python -m pytest tests/ -q
```

See [Commands](#commands) for the full CLI reference.

## What is gitignored and how to restore it

These paths are listed in [`.gitignore`](.gitignore). None of them should be
committed. After cloning the repo, work through [First-time local setup](#first-time-local-setup)
or use the table below to restore individual pieces.

| Gitignored path | How to recreate | Size (approx.) |
|-----------------|-----------------|----------------|
| `core_knowledge_stem/` | `python scripts/download_core_knowledge_stem.py` | ~935 MB |
| `models/bge-m3/` | `python scripts/download_retrieval_models.py` | ~2.3 GB |
| `models/bge-reranker-v2-m3/` | same script | ~2.3 GB |
| `models/qwen3-vl-8b-instruct/` | `huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b-instruct` | ~17 GB |
| `.neo4j-local/` | [Neo4j setup](#neo4j-setup): download Neo4j + JDK tarballs, extract, then `./scripts/neo4j_local.sh set-password` and `start` | ~500 MB install + DB grows with ingest |
| `.neo4j-local/*.tar.gz` | re-download the Neo4j/JDK tarballs from the URLs in [Neo4j setup](#neo4j-setup) | ~330 MB |
| `data/processed/` | `python scripts/ingest_corpus.py` | ~800 MB (mostly images) |
| `.env` | `cp .env.example .env` and set `NEO4J_PASSWORD` | — |
| `data/evaluation/retrieval_questions.jsonl` | hand-label per [Evaluation](#evaluation) | — |
| `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, `*.log` | recreated automatically by tools / re-running commands | negligible |

**Still committed:** source code, `scripts/`, `tests/`, `.env.example`,
`data/evaluation/README.md`, and `data/evaluation/retrieval_questions.example.jsonl`.

**If you wipe Neo4j data** (delete `.neo4j-local/neo4j-community-*/data/`),
re-run `init_neo4j.py` and `ingest_corpus.py`. The graph is not stored in git.

## Architecture

```
Core Knowledge PDFs
        |
   PyMuPDF structured parsing  (text, headings, embedded images, page geometry)
        |
   Hierarchical chunking       (Document -> Page -> Section -> Chunk)
        |
   BGE-M3 dense embeddings
        |
      Neo4j  ---- curriculum knowledge graph
             ---- dense vector index (cosine)
             ---- full-text (Lucene) index
             ---- metadata properties on Chunk nodes
        |
   Student query
        |
   Metadata filtering          (grade / subject / unit, supplied by the caller)
        |
   Three retrieval channels
        |-- dense semantic retrieval
        |-- lexical / full-text retrieval
        +-- bounded graph expansion
        |
   Weighted Reciprocal Rank Fusion
        |
   BGE-reranker-v2-m3
        |
   Evidence sufficiency gate
        |
   Qwen3-VL-8B-Instruct
        |
   Socratic tutoring response  (streamed)
```

Every layer is implemented directly. **LangChain and LlamaIndex are deliberately
not used**, because the research requires modifying and benchmarking chunking,
embedding, retrieval, graph traversal, fusion, reranking, evidence gating and
generation independently.

## Python version

**Python 3.12**

```bash
pip install -r requirements.txt
```

The curriculum downloader itself needs no third-party packages.

## Dataset

Curriculum PDFs come from the Core Knowledge Foundation:

<https://www.coreknowledge.org/download-free-curriculum/>

- Grade 1 Mathematics and Science
- Grade 2 Mathematics and Science
- Grade 3 Mathematics and Science

A full run downloads **51 units** and **126 PDFs** (~935 MB).

### Run the downloader

```bash
python scripts/download_core_knowledge_stem.py
```

Optional log:

```bash
python scripts/download_core_knowledge_stem.py 2>&1 | tee download.log
```

### What it does

1. Crawls the official curriculum listing for Grades 1–3 Mathematics and Science.
2. Visits each unit page and downloads every resource with a **Download Free PDF
   Version** link.
3. Saves files under `core_knowledge_stem/` using this layout:

```
core_knowledge_stem/
├── grade_01/
│   ├── mathematics/
│   └── science/
├── grade_02/
├── grade_03/
└── manifest.json
```

Each unit folder contains `student/`, `teacher/` and `other/` subdirectories
based on resource type.

4. Writes `core_knowledge_stem/manifest.json` with metadata for every PDF.
5. Re-crawls the site and verifies all expected units and PDFs exist locally.

The script skips PDFs that already exist, so it is safe to re-run after an
interruption.

### Important notes

- Materials are for **local research use only** and must not be redistributed.
- Only official Core Knowledge Foundation URLs are used; no login or paid
  resources are accessed.
- Raw PDFs are never modified. Ingestion writes derived artefacts to
  `data/processed/` and leaves `core_knowledge_stem/` untouched.
- Downloaded data is excluded from git via `.gitignore`.

## Models

All three models are stored locally and none are committed to the repository.

| Role | Model | Local path |
|------|-------|------------|
| Generator | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (original, non-quantized) | `models/qwen3-vl-8b-instruct` |
| Embeddings | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | `models/bge-m3` |
| Reranker | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `models/bge-reranker-v2-m3` |

### Generator

Qwen3-VL-8B-Instruct is loaded with Hugging Face Transformers using
`device_map="auto"`. Inference is slow. Streaming generation is supported and
is used by `scripts/test_rag.py`.

Transformers is installed from GitHub because Qwen3-VL support was not in a
tagged release at setup time. **Do not replace that dependency.**

Download the generator once:

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b-instruct
```

Download the retrieval models with:

```bash
python scripts/download_retrieval_models.py
```

### Embedding model

BGE-M3 produces the **dense** embeddings used for vector retrieval. Vectors are
L2-normalised so Neo4j cosine similarity is meaningful. The dimensionality is read
from the model's own config (`hidden_size`) and passed to the vector index; it is
never hardcoded.

Dense embeddings only are used today. The embedding wrapper is written so sparse
(lexical weights) and ColBERT multi-vector modes can be added later without
rewriting the retrieval pipeline.

### Reranker

BGE-reranker-v2-m3 is a cross-encoder scoring `(query, chunk_text)` pairs. It runs
on the top `fusion_top_k` fused candidates and returns the best `final_top_k`.

### Model loading

The three models are not all loaded at once. Each is loaded lazily and can be
released:

```
ingest: load BGE-M3 -> embed -> release BGE-M3
query : dense/lexical/graph retrieval uses stored vectors and indexes
        load reranker -> score -> (optionally) release
        load Qwen3-VL -> stream response
```

No quantization is used anywhere.

## Neo4j setup

Neo4j serves as **both** the graph database and the dense-vector store.
No separate vector database (Qdrant, Chroma, FAISS, Milvus, Weaviate) is used.

Neo4j runs **only** from a project-local tarball install under `.neo4j-local/`.
Use [`scripts/neo4j_local.sh`](scripts/neo4j_local.sh) to start, stop, and query
the database. There is no `docker-compose.yml` and no supported alternative
install path — every clone should follow the steps below.

Connection settings are read from environment variables, loaded from `.env`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt endpoint |
| `NEO4J_USER` | `neo4j` | Username |
| `NEO4J_PASSWORD` | *(required)* | Password |
| `NEO4J_DATABASE` | `neo4j` | Target database |

Copy the template and fill in a password:

```bash
cp .env.example .env
```

`.env` is git-ignored; `.env.example` contains no real credentials.

### Requirements

- Neo4j **5.13+** (native vector indexes); this project is verified on
  **Neo4j Community 5.26.12**
- **Java 17 or 21**. Neo4j 5.x does not run on newer JDKs.

### Tarball install under `.neo4j-local/`

Neo4j Community and a Temurin JDK 21 live under the git-ignored `.neo4j-local/`
directory. Every clone should provision the same layout so paths, scripts, and
the database location match:

```
.neo4j-local/
├── neo4j-community-5.26.12/     # extracted Neo4j + live data/
├── jdk-21.0.5+11/               # any jdk-21* directory name works
├── neo4j-community-5.26.12-unix.tar.gz   # optional after extract
└── temurin-jdk21.tar.gz                  # optional after extract
```

Provision it once:

```bash
mkdir -p .neo4j-local && cd .neo4j-local

# Neo4j Community 5.26.12 (needs 5.13+ for native vector indexes)
curl -LO https://dist.neo4j.org/neo4j-community-5.26.12-unix.tar.gz
tar --no-same-owner -xzf neo4j-community-5.26.12-unix.tar.gz

# Temurin JDK 21 (Linux x64). Any JDK 21 patch release works — do not use Java 22+.
# Option 1: download from https://adoptium.net/temurin/releases/?version=21
# Option 2: pick a GitHub release tarball, for example:
curl -LO https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz
tar --no-same-owner -xzf OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz

cd ..
./scripts/neo4j_local.sh set-password "<same-as-NEO4J_PASSWORD-in-.env>"
```

The tarball filenames are arbitrary (`temurin-jdk21.tar.gz` is fine). After
extraction, [`scripts/neo4j_local.sh`](scripts/neo4j_local.sh) finds the newest
`neo4j-community-*` and `jdk-21*` directories automatically.

The `.tar.gz` files are git-ignored and **safe to delete after extraction** (~330
MB saved). Deleting `.neo4j-local/neo4j-community-*/data/` wipes the graph; restore
it with `init_neo4j.py` and `ingest_corpus.py`.

Then manage the instance with the helper script, which pins `JAVA_HOME` to the
bundled JDK 21 so the system JDK is irrelevant:

```bash
./scripts/neo4j_local.sh start
./scripts/neo4j_local.sh status
./scripts/neo4j_local.sh stop
./scripts/neo4j_local.sh cypher -u neo4j -p "<your-password>"
```

Nothing is installed system-wide and no `sudo` is required.

## Graph schema

Structural hierarchy, derived entirely from document structure:

```
(:Grade)-[:HAS_SUBJECT]->(:Subject)
(:Subject)-[:HAS_UNIT]->(:Unit)
(:Unit)-[:HAS_DOCUMENT]->(:Document)
(:Document)-[:HAS_PAGE]->(:Page)
(:Page)-[:HAS_SECTION]->(:Section)
(:Section)-[:HAS_CHUNK]->(:Chunk)
(:Page)-[:HAS_IMAGE]->(:Image)
```

Additional supported edges:

```
(:Chunk)-[:MENTIONS]->(:Concept)      verbatim phrase or own-title evidence
(:Chunk)-[:ON_PAGE]->(:Page)          chunk to each page it covers
(:Chunk)-[:NEXT]->(:Chunk)            document reading order
(:Chunk)-[:PREVIOUS]->(:Chunk)        inverse of NEXT
(:Image)-[:APPEARS_IN]->(:Page)       inverse of HAS_IMAGE
(:Image)-[:ILLUSTRATES]->(:Concept)   only from text on the image's own page
```

`CAUSES`, `PREREQUISITE_OF`, `DEPENDS_ON`, `PROVES` and `IMPLIES` are
**intentionally absent**. They cannot be established reliably without model
inference, and letting an LLM invent graph facts would undermine the whole point
of grounding the tutor in verified curriculum.

### Identifiers

All IDs are deterministic, which is what makes ingestion idempotent:

```
grade_id    grade_01
subject_id  grade_01:science
unit_id     grade_01:science:unit_01_sun_moon_and_stars
document_id <unit_id>:student:student_reader
page_id     <document_id>:p0007
section_id  <document_id>:s0003
chunk_id    <section_id>:c0001
image_id    <page_id>:img01
concept_id  concept:plant_life_cycle
```

Re-running ingestion `MERGE`s the same nodes rather than duplicating them. A
document is re-processed only when its content hash, the ingest version or the
embedding version changes, or when `--force` is passed; its derived pages,
sections, chunks and images are then removed and rebuilt so no stale nodes
linger.

### Chunk metadata

Metadata is first-class: it lives on the `Chunk` node so filtering happens inside
the retrieval query, not after unrelated results have already been returned.

`chunk_id`, `text`, `token_count`, `chunk_index`, `section_id`, `section_title`,
`section_index`, `page_start`, `page_end`, `document_id`, `document_title`,
`unit_id`, `unit_title`, `subject_id`, `grade_id`, `grade`, `subject`,
`resource_type`, `audience`, `local_pdf_path`, plus `embedding`,
`embedding_version`, `embedding_model` and `embedding_dim`.

Every chunk is traceable up the full lineage: **Chunk → Section → Page →
Document → Unit → Subject → Grade**.

### Indexes

| Kind | Name | Detail |
|------|------|--------|
| Vector | `chunk_embedding_index` | `Chunk.embedding`, cosine, dimension read from BGE-M3 |
| Full-text | `chunk_fulltext_index` | `Chunk.text`, `Chunk.section_title`, `Chunk.unit_title` |
| Full-text | `concept_fulltext_index` | `Concept.name`, `Concept.normalized_name` |
| Uniqueness | 9 constraints | one per node label, on its ID property |
| Property | 11 indexes | grade+subject, unit, document, resource type, audience, embedding version, section, document hash, page/image document, normalized concept name |

### Concept extraction

Concepts are conservative by design. The vocabulary is built from **unit titles
and section titles across the whole corpus**, normalised for deduplication
(lowercasing, punctuation stripping, careful depluralisation), so "Plant Life
Cycle" and "plant life cycles" resolve to one concept. A `MENTIONS` edge is
created only when there is textual evidence: a verbatim phrase occurrence in the
chunk, or the chunk's own section/unit title. Every edge stores its evidence and
source. No LLM participates.

Single-word candidates supported by only one title are dropped, since those are
usually proper nouns from a story rather than recurring curriculum terms.

### Images

PyMuPDF extracts embedded raster images to
`data/processed/images/<grade>/<subject>/<unit>/`, using deterministic filenames.
`Image` nodes carry `image_id`, `local_path`, `source_pdf`, `page_number`,
`grade`, `subject`, `unit_id`, `document_id`, `width`, `height`, `format` and the
PDF `xref`.

Many diagrams in these PDFs are vector graphics, which cannot be extracted as
raster images. That is not an error: `Page` nodes always retain `source_pdf` and
`page_number`, so any page can be rendered on demand and handed to Qwen3-VL
later. No captions are generated and no captioning model is used.

## Retrieval pipeline

Everything below runs behind `HybridRetriever.retrieve(...)` in
`src/rag/pipeline.py`, but each stage is a separate module and each stage's output
is preserved.

### 1. Metadata filtering

```python
retriever.retrieve("why does the moon change shape", grade=1, subject="science")
```

`grade`, `subject`, `unit`, `resource_type`, `audience` and `document_id` are
optional filters. **Grade and subject are never inferred from the question text**
— the application already knows them, and guessing a student's grade would be
both unreliable and pedagogically wrong. With no filters, the whole corpus is
searched.

Filters become bound Cypher parameters applied during the index scan.

### 2. Dense semantic retrieval

Query → BGE-M3 dense embedding → `db.index.vector.queryNodes` → top `dense_top_k`.

Unfiltered queries use the approximate vector index directly. Filtered queries
over-fetch and then filter, falling back to an exact scan when the filtered scope
is small enough that the approximate index would return too few in-scope hits;
`diagnostics.notes` records which strategy ran.

### 3. Lexical / full-text retrieval

Query → Lucene full-text index over `Chunk.text` + section/unit titles → top
`fulltext_top_k`. This recovers exact terminology, names, curriculum phrases and
numbers that dense retrieval blurs. Query text is escaped and stop-worded before
being handed to Lucene, and the resulting query string is recorded in the
diagnostics.

### 4. Graph expansion

Seeded by the strongest dense and lexical hits (`graph_seed_top_k`), never by
blind traversal. Bounded expansion over: same section, `NEXT`/`PREVIOUS`
neighbours, same page, shared `Concept`, and same unit. Depth is capped by
`GRAPH_MAX_DEPTH` (default 2) and total candidates by `GRAPH_TOP_K`.

Each graph candidate records `graph_seed_chunk_id` and `graph_expansion_path`, so
it stays attributable to the curriculum structure that produced it.

### 5. Weighted Reciprocal Rank Fusion

Raw channel scores are never summed: a cosine similarity, a Lucene BM25-style
score and a graph weight live on different scales. RRF uses ranks only:

```
score(chunk) = Σ_channels  weight_channel / (k + rank_channel)
```

`k` defaults to 60. Dense and full-text are primary signals; graph expansion is
secondary (lower weight). Fusion deduplicates, and a chunk found by several
channels lists all of them in `retrieval_sources` with per-channel RRF
contributions retained.

### 6. Reranking

The top `fusion_top_k` (default 20) fused candidates are scored by
BGE-reranker-v2-m3 and the best `final_top_k` (default 5) are returned.

### Preserved diagnostics

`RetrievalResponse.diagnostics` retains, for every query: dense candidates with
ranks and scores, full-text candidates with ranks and scores, graph candidates
with expansion paths and seeds, the graph seed list, the fused ranking with RRF
scores and per-channel contributions, reranker scores, per-stage timings, and
notes about which dense strategy and Lucene query were used.

Nothing is discarded, because later work compares retrieval architectures
experimentally.

## Evidence sufficiency gate

`src/rag/evidence.py` decides whether the retrieved evidence justifies answering
at all. It is a separate component precisely so it can be replaced by a stronger
grounding check later.

The gate currently considers: whether any candidates exist; the top reranker
score; how many chunks clear a "strong" threshold; whether evidence actually
comes from the requested grade and subject; and whether the retrieved text
overlaps the question's content terms.

**These thresholds are heuristics and are not validated.**
They are configurable (`EVIDENCE_*` variables) and are meant to be tuned against
a real labelled set.

When evidence is insufficient the pipeline returns a structured
insufficient-evidence result and the generator is asked to say that verified
curriculum evidence is insufficient — it is never asked to invent an answer.

## Socratic controller

`src/rag/socratic.py` owns all tutoring behaviour, kept separate from retrieval so
prompt changes do not touch the retrieval code.

The controller composes the system prompt from the student's grade and subject,
formats retrieved evidence with its provenance, and supports tutoring states:
`ASK_QUESTION`, `GIVE_HINT`, `CORRECT_MISCONCEPTION`, `EXPLAIN_CONCEPT`,
`CONFIRM_STEP` and `INSUFFICIENT_EVIDENCE`. Student state is not modelled across
turns, but the interface takes a conversation history so it does not assume
one-shot answers.

The generator is instructed to treat retrieved curriculum as the factual basis,
not to fabricate curriculum facts, not to reveal the full solution immediately, to
guide with questions and small steps, to match the student's grade, and to admit
insufficient evidence rather than guess. Source metadata (document, page, unit,
chunk) is preserved internally so citations can be surfaced later; no URLs are
fabricated.

## Commands

Run all commands from the repository root.

### 1. Download the retrieval models

```bash
python scripts/download_retrieval_models.py
python scripts/download_retrieval_models.py --model bge-m3
python scripts/download_retrieval_models.py --force
```

Downloads BGE-M3 and BGE-reranker-v2-m3 only if missing. Already-complete
models are skipped and reported in a status table at the end. Never touches
Qwen3-VL. Flags: `--model` (`bge-m3` or `bge-reranker-v2-m3`, repeatable),
`--force` (re-download even when complete).

### 2. Start Neo4j

Requires a provisioned `.neo4j-local/` (see [Neo4j setup](#neo4j-setup)).

```bash
./scripts/neo4j_local.sh start
```

### 3. Initialise the schema

```bash
python scripts/init_neo4j.py
```

Creates constraints, property indexes, the vector index (dimension read from
BGE-M3) and the full-text indexes. Idempotent. Flags: `--show` (report schema and
counts only), `--reset` (delete all nodes; prompts unless `--yes`), `--drop-indexes`
(with `--reset`, also drop indexes).

### 4. Ingest the corpus

```bash
python scripts/ingest_corpus.py
```

Parses PDFs, builds hierarchical chunks, extracts images, upserts the graph,
generates embeddings and links concepts. Safe and idempotent by default:
up-to-date documents are skipped, so an interrupted run resumes.

```bash
python scripts/ingest_corpus.py --grade 1 --subject science   # subset
python scripts/ingest_corpus.py --limit 5                     # smoke test
python scripts/ingest_corpus.py --force                       # re-process all
python scripts/ingest_corpus.py --skip-embeddings             # graph only
python scripts/ingest_corpus.py --keep-embedder               # leave BGE-M3 loaded
python scripts/ingest_corpus.py --reset                       # destructive, prompts
python scripts/ingest_corpus.py --reset --yes                 # skip prompt
```

### 5. Inspect the graph

```bash
python scripts/inspect_graph.py
python scripts/inspect_graph.py --units
python scripts/inspect_graph.py --unit grade_01:science:unit_01_sun_moon_and_stars
python scripts/inspect_graph.py --chunk "<chunk_id>"
python scripts/inspect_graph.py --concepts 30
python scripts/inspect_graph.py --images 10
```

### 6. Test retrieval (no generator)

```bash
python scripts/test_retriever.py --query "why does the moon change shape" --grade 1 --subject science
```

Prints dense, full-text, graph, fused and reranked results side by side with per
channel ranks, scores, RRF scores and reranker scores. This script never calls
the generator. Useful flags: `-q`/`--query`, `-g`/`--grade`, `-s`/`--subject`,
`-u`/`--unit`, `--resource-type`, `--audience`, `--limit`, `--final-top-k`,
`--no-rerank`, `--images`, `--json`.

### 7. Test the full RAG pipeline

```bash
python scripts/test_rag.py --query "why does the moon look different on different nights" --grade 1 --subject science
python scripts/test_rag.py --query "why does the moon look different on different nights" --grade 1 --subject science --state GIVE_HINT
python scripts/test_rag.py --query "what is a black hole" --grade 1 --subject science --strict
python scripts/test_rag.py --query "measuring length" --grade 2 --subject mathematics --retrieval-only
```

Runs filtering, all three channels, fusion, reranking and the evidence gate,
prints concise diagnostics, and only then streams the Socratic response from
Qwen3-VL. If the gate reports insufficient evidence, the generator is either
asked to decline (`--strict` off) or not loaded at all (`--strict` on).

Omit `-q`/`--query` (and optionally `-g`/`-s`) to be prompted interactively.

#### CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-q`, `--query` | *(prompt)* | Student question |
| `-g`, `--grade` | any | Grade filter (`1`, `2`, or `3`) |
| `-s`, `--subject` | any | `science` or `mathematics` |
| `-u`, `--unit` | any | Restrict retrieval to one unit id |
| `--state` | `ASK_QUESTION` | Tutoring move (see table below) |
| `--retrieval-only` | off | Stop after the evidence gate; never load the generator |
| `--strict` | off | On insufficient evidence, skip generation entirely |
| `--max-new-tokens` | from config (`640`) | Cap streamed answer length for this run |
| `--json PATH` | — | Write the full result record (diagnostics, gate, response) to a file |
| `--log-level` | `WARNING` | Logging verbosity |

#### `--state` tutoring moves

These change how the generator is prompted. They do **not** bypass the evidence
gate, and there is no mode that reveals the full solution outright — every state
is Socratic.

| Value | Behaviour |
|-------|-----------|
| `ASK_QUESTION` | Default. One short question to start thinking; no final answer. |
| `GIVE_HINT` | One small hint, then ask the student to try that step. |
| `CORRECT_MISCONCEPTION` | Acknowledge what is right; question the wrong assumption; correct from evidence only. |
| `EXPLAIN_CONCEPT` | Two or three short sentences from the evidence, then a check question. |
| `CONFIRM_STEP` | Confirm whether the student's step is correct and why, then ask about the next step. |

`INSUFFICIENT_EVIDENCE` is chosen automatically when the gate fails — it is not a
valid `--state` value. With `--strict`, generation is skipped instead of running
in that state.

### 8. Generator sanity check (pre-existing)

```bash
python scripts/test_generator.py
```

### 9. Evaluate retrieval (needs manual labels)

```bash
python scripts/evaluate_retrieval.py --per-question
python scripts/evaluate_retrieval.py --questions data/evaluation/retrieval_questions.jsonl
python scripts/evaluate_retrieval.py --no-rerank --json report.json
```

Flags: `--questions`, `--k`, `--no-rerank`, `--per-question`, `--json`.

## Configuration

All tunables live in `src/rag/config.py` and are overridable via environment
variables or `.env`. There are no magic numbers scattered through the code.

| Variable | Default | Meaning |
|----------|---------|---------|
| `CORPUS_PATH` | `core_knowledge_stem` | Raw PDF corpus root |
| `PROCESSED_DATA_PATH` | `data/processed` | Derived artefacts |
| `EMBEDDING_MODEL_PATH` | `models/bge-m3` | BGE-M3 directory |
| `RERANKER_MODEL_PATH` | `models/bge-reranker-v2-m3` | Reranker directory |
| `GENERATOR_MODEL_PATH` | `models/qwen3-vl-8b-instruct` | Qwen3-VL directory |
| `CHUNK_TARGET_TOKENS` | `600` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `100` | Overlap between chunks |
| `CHUNK_MIN_TOKENS` | `80` | Below this a chunk is merged forward |
| `CHUNK_MAX_TOKENS` | `900` | Hard ceiling |
| `INGEST_MIN_SECTION_CHARS` | `1500` | Short heading-delimited sections merge into the previous one |
| `DENSE_TOP_K` | `20` | Dense candidates |
| `FULLTEXT_TOP_K` | `20` | Lexical candidates |
| `GRAPH_SEED_TOP_K` | `10` | Seeds for graph expansion |
| `GRAPH_MAX_DEPTH` | `2` | Traversal depth cap (1–3) |
| `GRAPH_TOP_K` | `20` | Graph candidate cap, so traversal cannot flood fusion |
| `FUSION_TOP_K` | `20` | Candidates entering the reranker |
| `FINAL_TOP_K` | `5` | Chunks returned as evidence |
| `RRF_K` | `60` | RRF constant |
| `WEIGHT_DENSE` | `1.0` | Dense channel weight |
| `WEIGHT_FULLTEXT` | `1.0` | Lexical channel weight |
| `WEIGHT_GRAPH` | `0.5` | Graph channel weight (secondary) |
| `EVIDENCE_MIN_RERANK_SCORE` | `0.0` | Minimum top reranker logit (0 ≈ sigmoid 0.5) |
| `EVIDENCE_MIN_CHUNKS` | `1` | Minimum candidates required |
| `EVIDENCE_MIN_STRONG_CHUNKS` | `1` | Chunks that must clear the score threshold |
| `EVIDENCE_REQUIRE_SCOPE_MATCH` | `true` | Evidence must come from the requested grade/subject |
| `EVIDENCE_MIN_QUERY_TERM_OVERLAP` | `0.15` | Fraction of query content words that must appear |
| `MULTIMODAL_ENABLED` | `false` | Surface retrieved images alongside text |
| `EMBEDDING_BATCH_SIZE` | `8` | Embedding batch size |
| `EMBEDDING_DEVICE` | `auto` | `cuda`, `cpu`, or auto-detect |
| `RERANKER_BATCH_SIZE` | `4` | Reranker batch size |
| `INGEST_EXTRACT_IMAGES` | `true` | Extract embedded raster images |
| `INGEST_ENABLE_OCR` | `false` | OCR is off; text-layer extraction only |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

See `.env.example` for the full annotated list.

## Evaluation

`data/evaluation/` holds manually curated retrieval questions. **No labels ship
with this repository**: inventing relevance judgements to make metrics run would
produce meaningless numbers.

`data/evaluation/README.md` documents the JSONL schema (`question`, `grade`,
`subject`, `expected_unit`, `relevant_chunk_ids`, `relevance_grades`, `notes`) and
the labelling procedure. `retrieval_questions.example.jsonl` shows two schema
examples with empty label lists.

`scripts/evaluate_retrieval.py` computes Recall@K, MRR, nDCG@K and unit hit
rate@K — for fused and reranked output separately — but only for questions that
carry the labels each metric needs. Unlabelled questions are reported as skipped,
never scored as zero.

## Tests

```bash
python -m pytest tests/ -q
```

| File | Scope | Needs Neo4j? |
|------|-------|--------------|
| `tests/test_chunking.py` | hierarchy, overlap, token budgets, deterministic IDs | no |
| `tests/test_metadata.py` | ID scheme, filter → Cypher translation, concept normalisation | no |
| `tests/test_fusion.py` | RRF arithmetic, dedup, weighting, signal preservation | no |
| `tests/test_retrieval.py` | live channels, metadata isolation, traceability | yes (skips if absent) |

`tests/test_retrieval.py` skips rather than fails when Neo4j, an ingested corpus
or the BGE models are unavailable.

## Project structure

```
sih-stem-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   └── rag/
│       ├── config.py           centralised configuration
│       ├── schemas.py          dataclasses + deterministic ID scheme
│       ├── logging_utils.py    structured logging, timers
│       ├── corpus.py           PDF discovery + manifest metadata
│       ├── pdf_parser.py       PyMuPDF text/section/image extraction
│       ├── chunker.py          hierarchical token-aware chunking
│       ├── concepts.py         conservative concept extraction
│       ├── embeddings.py       BGE-M3 dense embeddings
│       ├── reranker.py         BGE-reranker-v2-m3
│       ├── neo4j_store.py      driver wrapper, error translation
│       ├── graph_schema.py     labels, constraints, vector/full-text indexes
│       ├── ingest.py           idempotent ingestion
│       ├── retrieval_base.py   shared projection + filter builder
│       ├── dense_retriever.py  vector channel
│       ├── lexical_retriever.py full-text channel
│       ├── graph_retriever.py  bounded graph expansion
│       ├── fusion.py           weighted RRF
│       ├── evidence.py         evidence sufficiency gate
│       ├── socratic.py         tutoring controller and prompts
│       ├── generator.py        Qwen3-VL wrapper with streaming
│       └── pipeline.py         HybridRetriever + SocraticRagPipeline
├── scripts/
│   ├── download_core_knowledge_stem.py
│   ├── download_retrieval_models.py
│   ├── neo4j_local.sh
│   ├── init_neo4j.py
│   ├── ingest_corpus.py
│   ├── inspect_graph.py
│   ├── test_retriever.py
│   ├── test_rag.py
│   ├── test_generator.py
│   └── evaluate_retrieval.py
├── tests/
├── data/
│   ├── processed/              text/, images/, manifests/, cache/  (git-ignored)
│   └── evaluation/             JSONL schema + labelling guide
├── models/                     git-ignored
│   ├── qwen3-vl-8b-instruct/
│   ├── bge-m3/
│   └── bge-reranker-v2-m3/
├── .neo4j-local/               git-ignored Neo4j tarball + JDK 21 + DB data
└── core_knowledge_stem/        git-ignored raw PDFs
```

There is no `docker-compose.yml`. Neo4j is started with `scripts/neo4j_local.sh`
after provisioning `.neo4j-local/` as described in [Neo4j setup](#neo4j-setup).

## Current limitations

- **BGE-M3 sparse (lexical-weight) retrieval is not implemented.** Lexical
  matching comes from the Neo4j full-text index instead.
- **ColBERT / multi-vector late-interaction retrieval is not implemented.**
- **Multimodal embeddings are not implemented.** No image or page is embedded.
- **Images are stored and referenced through graph paths only.** `Image` nodes
  carry file paths and page provenance; retrieval reaches them structurally, not
  visually. Passing a rendered page to Qwen3-VL is possible but off by default.
- **Concept relationships are intentionally conservative.** Only `MENTIONS` and
  `ILLUSTRATES`, both requiring textual evidence. No inferred semantic edges.
- **Full student-state modelling is not implemented.** The Socratic controller
  supports tutoring states but nothing tracks a student's mastery across turns.
- **Evidence-gate thresholds are unvalidated heuristics** and need tuning against
  a labelled set.
- **No retrieval accuracy numbers exist yet**, because no relevance labels have
  been created.
- **Section titles are noisy.** Detection relies on typographic heuristics, so
  some sections are labelled with page furniture such as `ISBN: 978-1-68380-585-4`,
  `CHAPTER` or `AP 3.3.1` instead of a real lesson heading. The chunk *text* and
  the page/document/unit lineage are unaffected, and retrieval works, but section
  titles are not reliable display labels yet.
- Teacher guides with dense sub-headings produce many more sections than student
  readers, and short headings are folded into body text when a section is too
  small to stand alone.
- Chunk-level overlap is text-based when a paragraph exceeds the overlap budget,
  so the shared span is not always a whole paragraph.
- OCR is not used. Pages with no text layer are recorded as image-only and
  contribute no chunks.
- Generation can be slow.

## Future experimental directions

All of the following are **future work, not implemented**:

- BGE-M3 sparse retrieval
- dense + sparse hybrid search
- ColBERT / multi-vector late interaction
- stronger graph relationship extraction
- parent-child retrieval comparison (returning parent sections instead of chunks)
- query rewriting and decomposition
- multimodal visual retrieval (embedding diagrams and pages)
- stronger grounding verification to replace the current evidence gate
