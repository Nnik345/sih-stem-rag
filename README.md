# SIH STEM RAG

Local research project building a **hallucination-resistant Socratic STEM tutor**
over a CISCE-aligned Grade 3–5 STEM corpus (EngageNY Mathematics, Siyavula
Natural Sciences, and Utah Science OER). CISCE is used only as an alignment
authority and is never ingested.

The system implements hybrid GraphRAG retrieval on a local Neo4j instance, feeding a
locally hosted Qwen3-VL-8B-Instruct generator that is instructed to tutor rather
than to answer outright.

## Contents

- [Complete setup and run](#complete-setup-and-run)
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
- [Local observability dashboard](#local-observability-dashboard)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Project structure](#project-structure)
- [Current limitations](#current-limitations)
- [Future experimental directions](#future-experimental-directions)

## Complete setup and run

Follow these commands in order from a fresh clone. Everything is local: Neo4j
under `.neo4j-local/`, models under `models/`, and the dashboard on
`127.0.0.1`. Use the active Python 3.12 environment. Node.js is required only
for the dashboard (check with `node --version` and `npm --version`).

### 1. Python packages

```bash
pip install -r requirements.txt
```

This installs the existing RAG stack plus the visualizer API (`fastapi`,
`uvicorn`, `httpx`) and `pytest`.

### 2. Environment file

```bash
cp .env.example .env
```

Set `NEO4J_PASSWORD` in `.env` to a real password (not `change-me`).

### 3. Curriculum PDFs

EngageNY Grade 3–5 Mathematics **must be downloaded by hand** (NYSED SharePoint
requires a Microsoft login). Then run the automated downloader for Siyavula,
Utah, and the CISCE alignment PDF. See [Dataset](#dataset).

```bash
python scripts/download_curriculum.py
```

### 4. Models

```bash
python scripts/download_retrieval_models.py
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir models/qwen3-vl-8b-instruct
```

### 5. Neo4j (tarball under `.neo4j-local/` only)

If `.neo4j-local/` is not already provisioned, follow [Neo4j setup](#neo4j-setup)
(download the Neo4j Community and Temurin JDK 21 tarballs, extract them, then
set the password). Then:

```bash
./scripts/neo4j_local.sh set-password "<same-as-NEO4J_PASSWORD-in-.env>"
./scripts/neo4j_local.sh start
python scripts/init_neo4j.py
python scripts/ingest_corpus.py
```

Skip `set-password` if the database already has this password. Re-running
`ingest_corpus.py` is idempotent.

### 6. Dashboard frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

`npm install` fills gitignored `frontend/node_modules/` from the committed
`frontend/package-lock.json`. `npm run build` writes gitignored `frontend/dist/`.
Do not install npm packages globally.

### 7. Run the visualizer

```bash
python scripts/run_visualizer.py
```

The browser opens `http://127.0.0.1:8000`. Submit a query such as
`how does weather change` with grade `3` and
subject `science`. Use retrieval-only first if you do not want to load Qwen.

Optional launcher flags: `--host`, `--port`, `--reload`, `--no-browser`.

Frontend hot-reload (API still on port 8000):

```bash
python scripts/run_visualizer.py --no-browser
# in another terminal:
cd frontend
npm run dev
```

Then open `http://127.0.0.1:5173`.

### 8. CLI pipeline (without the dashboard)

Neo4j must still be running (`./scripts/neo4j_local.sh start`).

```bash
python scripts/test_retriever.py \
    --query "what is a unit fraction" \
    --grade 3 --subject mathematics

python scripts/test_rag.py \
    --query "how does weather change" \
    --grade 3 --subject science

python scripts/test_rag.py \
    --query "how do plants make food" \
    --grade 4 --subject science --retrieval-only
```

### 9. Tests

```bash
python -m pytest tests/ -q
cd frontend
npm test -- --run
```

Integration tests skip (they do not fail) when Neo4j or local models are missing.

## First-time local setup

Everything runs locally. Neo4j is installed from tarballs under `.neo4j-local/`
and managed with [`scripts/neo4j_local.sh`](scripts/neo4j_local.sh) — no system
packages, no root, and no Docker. Large downloads and generated data are
git-ignored; see the next section for how to restore them after a fresh clone.

**Prerequisites:** Python **3.12** and ~25 GB free disk for models, corpus, and
Neo4j (more during ingestion).

### 1. Python environment

```bash
pip install -r requirements.txt
```

### 2. Environment file

```bash
cp .env.example .env
```

Edit `.env` and set `NEO4J_PASSWORD` to a real password. The same value is used by
Neo4j and by every script that connects to the database.

### 3. Curriculum sources

EngageNY Mathematics Grade 3–5 full-module PDFs **cannot be fetched by the
automated downloader**. NYSED hosts them behind SharePoint (Microsoft login).
Placing those 20 PDFs is a required setup step, not optional.

1. Open an official NYSED page and sign in if prompted:
   - https://www.nysed.gov/edtech/digital-content-resources-mathematics
   - https://www.nysed.gov/standards-instruction/standards-resources-and-supports
   - SharePoint folder: https://nysed.sharepoint.com/:f:/s/P12EngageNY-Math-EXTA/En7SIs8H6v5PlQbP8fYWQbkBvFl7pdadxm5WQe2RYn6C_Q?e=aA13JQ
2. Download Grade 3, Grade 4 and Grade 5 Mathematics. Keep only
   `math-gN-mN-full-module.pdf` (20 files: G3 M1–7, G4 M1–7, G5 M1–6). Skip
   assessments, answer keys, lesson zips and Internet Archive copies.
3. Copy each PDF to the catalog path, for example:

```
curriculum/raw/engageny/mathematics/grade_03/module_01_properties_of_multiplication_and_division/student/math-g3-m1-full-module.pdf
```

Expected filenames are `math-g{3,4,5}-m{1…}-full-module.pdf` under the matching
`grade_XX/module_…/student/` directory listed in
`src/rag/curriculum_catalog.py`.

4. Then fetch everything the script *can* download (Siyavula, Utah, CISCE
   alignment PDF). Existing valid files are skipped:

```bash
python scripts/download_curriculum.py
```

Output: `curriculum/raw/` (git-ignored). Details in [Dataset](#dataset).

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
python scripts/ingest_corpus.py    # ~8 minutes for 25 catalog files
```

Ingestion writes git-ignored artefacts under `data/processed/` (images,
manifests, cache) and populates Neo4j. Re-running is idempotent unless you pass
`--force`.

### 8. Smoke verification

```bash
python scripts/test_retriever.py \
    --query "what is a unit fraction" \
    --grade 3 --subject mathematics

python scripts/test_rag.py \
    --query "how does weather change" \
    --grade 3 --subject science

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
| `curriculum/raw/` | Manual EngageNY full-module PDFs (see [Dataset](#dataset)), then `python scripts/download_curriculum.py` | several hundred MB |
| `models/bge-m3/` | `python scripts/download_retrieval_models.py` | ~2.3 GB |
| `models/bge-reranker-v2-m3/` | same script | ~2.3 GB |
| `models/qwen3-vl-8b-instruct/` | `huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b-instruct` | ~17 GB |
| `.neo4j-local/` | [Neo4j setup](#neo4j-setup): download Neo4j + JDK tarballs, extract, then `./scripts/neo4j_local.sh set-password` and `start` | ~500 MB install + DB grows with ingest |
| `.neo4j-local/*.tar.gz` | re-download the Neo4j/JDK tarballs from the URLs in [Neo4j setup](#neo4j-setup) | ~330 MB |
| `curriculum/processed/` | `python scripts/ingest_corpus.py` | grows with ingest |
| `.env` | `cp .env.example .env` and set `NEO4J_PASSWORD` | — |
| `data/evaluation/retrieval_questions.jsonl` | hand-label per [Evaluation](#evaluation) | — |
| `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, `*.log` | recreated automatically by tools / re-running commands | negligible |
| `frontend/node_modules/` | `cd frontend && npm install` | depends on lockfile |
| `frontend/dist/` | `cd frontend && npm run build` | small |
| `frontend/.vite/`, `frontend/.pytest_cache/`, `frontend/coverage/` | recreated by Vite / accidental pytest in `frontend/` | negligible |
| `.tools/` | optional local Node.js LTS archive if system Node is unavailable | ~50 MB |

**Still committed:** source code, `scripts/`, `tests/`, `frontend/` (except
`node_modules/` and `dist/`), `.env.example`,
`data/evaluation/README.md`, and `data/evaluation/retrieval_questions.example.jsonl`.

**If you wipe Neo4j data** (delete `.neo4j-local/neo4j-community-*/data/`),
re-run `init_neo4j.py` and `ingest_corpus.py`. The graph is not stored in git.

## Architecture

```
Approved STEM sources (PDF/ePUB)
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

Supported grades are **3, 4 and 5 only**. CISCE is the alignment authority and is
**not ingested**, embedded, or retrieved.

| Source | Grades | Subject | Role | Licence |
| ------ | -----: | ------- | ---- | ------- |
| CISCE Primary Curriculum | 3–5 | Mathematics and Science | Alignment only | CISCE terms; PDF not in Neo4j |
| EngageNY Mathematics | 3–5 | Mathematics | Primary | CC BY-NC-SA |
| Siyavula Natural Sciences (learner ePUB, CC-BY) | 4–5 | Science | Primary | CC BY |
| Utah Science OER | 3 | Science | Primary | Mixed OER notices (may include noncommercial terms) |
| Utah Science OER | 4–5 | Science | Support | Mixed OER notices (may include noncommercial terms) |

Grade 3 Science uses Utah OER as primary because Siyavula begins at Grade 4.
EngageNY teacher solutions and answer keys are classified `evaluation_only` and
never enter production retrieval. Noncommercial restrictions apply to EngageNY
and may apply to portions of Utah content.

### Run the downloader

```bash
python scripts/download_curriculum.py
```

Only the URLs in `src/rag/curriculum_catalog.py` are used. There is no login and
no mirror fallback. Valid files are skipped on re-run. SHA-256 hashes are
written to `curriculum/manifests/sources.yaml` and
`curriculum/manifests/checksums.sha256`.

**EngageNY must be downloaded manually.** NYSED public pages point at a
SharePoint folder that requires a Microsoft login, so
`scripts/download_curriculum.py` will not retrieve those PDFs. If the 20
full-module files are already in `curriculum/raw/engageny/`, the script skips
them. If they are missing, it tells you where to put them.

Required files (CC BY-NC-SA; local research only):

| Grade | Modules | Filename pattern |
| ----- | ------: | ---------------- |
| 3 | 1–7 | `math-g3-mN-full-module.pdf` |
| 4 | 1–7 | `math-g4-mN-full-module.pdf` |
| 5 | 1–6 | `math-g5-mN-full-module.pdf` |

Official starting pages:

- https://www.nysed.gov/edtech/digital-content-resources-mathematics
- https://www.nysed.gov/standards-instruction/standards-resources-and-supports
- https://www.nysed.gov/curriculum-instruction/engageny
- SharePoint: https://nysed.sharepoint.com/:f:/s/P12EngageNY-Math-EXTA/En7SIs8H6v5PlQbP8fYWQbkBvFl7pdadxm5WQe2RYn6C_Q?e=aA13JQ

Do not use Internet Archive or other unofficial mirrors. After copying the PDFs
into the catalog paths, re-run the downloader (to record hashes) and ingest:

```bash
python scripts/download_curriculum.py
python scripts/ingest_corpus.py
```

Layout:

```
curriculum/
├── manifests/sources.yaml
├── alignment/cisce_grade_3_5_stem.yaml
├── raw/          git-ignored downloads
└── processed/    git-ignored ingest artefacts
```

The CISCE PDF is stored under `curriculum/raw/_alignment_only/` for local
alignment authoring and is never ingested.

### Replace the old graph, then ingest

```bash
python scripts/replace_corpus.py --purge-core-knowledge --yes
python scripts/replace_corpus.py --purge-all-curriculum --yes   # explicit full wipe
python scripts/ingest_corpus.py
```

Ordinary ingestion does not delete the graph.

### Important notes

- Materials are for **local research use only**.
- EngageNY is CC BY-NC-SA; do not use it commercially.
- Utah textbooks mix CK-12 and other notices; images without a clear licence
  are skipped.
- Siyavula learner ePUBs are CC BY; branded ND PDFs are not used.
- All CISCE mappings are `needs_human_review` until a reviewer signs them.

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
grade_id    grade_03
subject_id  grade_03:science
unit_id     grade_03:science:unit_01_science_oer
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
retriever.retrieve("what is a unit fraction", grade=3, subject="mathematics")
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

### 1. Download curriculum sources

EngageNY Grade 3–5 full-module PDFs must be copied in by hand first (SharePoint
login). Paths and official pages are in [Dataset](#dataset). Then:

```bash
python scripts/download_curriculum.py
```

Fetches Siyavula, Utah, and the CISCE alignment PDF. Existing valid files,
including local EngageNY PDFs, are skipped. If EngageNY files are missing, the
script prints the expected destination paths and exits with an error.

### 2. Download the retrieval models

```bash
python scripts/download_retrieval_models.py
python scripts/download_retrieval_models.py --model bge-m3
python scripts/download_retrieval_models.py --force
```

Downloads BGE-M3 and BGE-reranker-v2-m3 only if missing. Already-complete
models are skipped and reported in a status table at the end. Never touches
Qwen3-VL. Flags: `--model` (`bge-m3` or `bge-reranker-v2-m3`, repeatable),
`--force` (re-download even when complete).

### 3. Start Neo4j

Requires a provisioned `.neo4j-local/` (see [Neo4j setup](#neo4j-setup)).

```bash
./scripts/neo4j_local.sh start
```

### 4. Initialise the schema

```bash
python scripts/init_neo4j.py
```

Creates constraints, property indexes, the vector index (dimension read from
BGE-M3) and the full-text indexes. Idempotent. Flags: `--show` (report schema and
counts only), `--reset` (delete all nodes; prompts unless `--yes`), `--drop-indexes`
(with `--reset`, also drop indexes).

### 5. Ingest the corpus

```bash
python scripts/ingest_corpus.py
```

Parses PDFs, builds hierarchical chunks, extracts images, upserts the graph,
generates embeddings and links concepts. Safe and idempotent by default:
up-to-date documents are skipped, so an interrupted run resumes.

```bash
    python scripts/ingest_corpus.py --grade 3 --subject science   # subset
python scripts/ingest_corpus.py --limit 5                     # smoke test
python scripts/ingest_corpus.py --force                       # re-process all
python scripts/ingest_corpus.py --skip-embeddings             # graph only
python scripts/ingest_corpus.py --keep-embedder               # leave BGE-M3 loaded
python scripts/ingest_corpus.py --reset                       # destructive, prompts
python scripts/ingest_corpus.py --reset --yes                 # skip prompt
```

### 6. Inspect the graph

```bash
python scripts/inspect_graph.py
python scripts/inspect_graph.py --units
python scripts/inspect_graph.py --unit grade_03:science:unit_01_science_oer
python scripts/inspect_graph.py --chunk "<chunk_id>"
python scripts/inspect_graph.py --concepts 30
python scripts/inspect_graph.py --images 10
```

### 7. Test retrieval (no generator)

```bash
python scripts/test_retriever.py --query "what is a unit fraction" --grade 3 --subject mathematics
```

Prints dense, full-text, graph, fused and reranked results side by side with per
channel ranks, scores, RRF scores and reranker scores. This script never calls
the generator. Useful flags: `-q`/`--query`, `-g`/`--grade`, `-s`/`--subject`,
`-u`/`--unit`, `--resource-type`, `--audience`, `--limit`, `--final-top-k`,
`--no-rerank`, `--images`, `--json`.

### 8. Test the full RAG pipeline

```bash
python scripts/test_rag.py --query "how does weather change" --grade 3 --subject science
python scripts/test_rag.py --query "how does weather change" --grade 3 --subject science --state GIVE_HINT
python scripts/test_rag.py --query "what is a unit fraction" --grade 3 --subject mathematics --strict
python scripts/test_rag.py --query "how do I find the area of a rectangle" --grade 3 --subject mathematics --retrieval-only
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
| `-g`, `--grade` | any | Grade filter (`3`, `4`, or `5`) |
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

### 9. Generator sanity check (pre-existing)

```bash
python scripts/test_generator.py
```

### 10. Evaluate retrieval (needs manual labels)

```bash
python scripts/evaluate_retrieval.py --per-question
python scripts/evaluate_retrieval.py --questions data/evaluation/retrieval_questions.jsonl
python scripts/evaluate_retrieval.py --no-rerank --json report.json
```

Flags: `--questions`, `--k`, `--no-rerank`, `--per-question`, `--json`.

## Local observability dashboard

A local browser dashboard shows how one query moves through retrieval, fusion,
reranking, the evidence gate, the Socratic prompt and generation. Graph expansion
includes nodes that were examined and then ignored, with machine-readable reason
codes. Tracing is optional: the CLI pipeline is unchanged when no observer is
attached, and diagnostic graph queries run only while the dashboard is tracing a
run.

Everything stays on this machine. The server binds to `127.0.0.1` by default.
There is no cloud service, CDN, telemetry, or analytics.

### Setup and launch

From the repository root, using the active Python environment:

```bash
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
python scripts/run_visualizer.py
```

The launcher opens `http://127.0.0.1:8000` unless you pass `--no-browser`.
Optional flags: `--host`, `--port`, `--reload`, `--no-browser`. If
`frontend/dist/` is missing, the script prints the build commands above and
exits without starting a server.

Frontend packages stay in `frontend/node_modules/`. Do not install npm packages
globally.

### Frontend development

Run the API and the Vite dev server separately. Vite proxies `/api` to port 8000
and is allowed only from the local origin.

```bash
python scripts/run_visualizer.py --no-browser
```

In another terminal:

```bash
cd frontend
npm run dev
```

Then open `http://127.0.0.1:5173`.

## Configuration

All tunables live in `src/rag/config.py` and are overridable via environment
variables or `.env`. There are no magic numbers scattered through the code.

| Variable | Default | Meaning |
|----------|---------|---------|
| `CORPUS_PATH` | `curriculum` | Curriculum root (raw + manifests) |
| `PROCESSED_DATA_PATH` | `curriculum/processed` | Derived artefacts |
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
cd frontend && npm test -- --run
```

| File | Scope | Needs Neo4j? |
|------|-------|--------------|
| `tests/test_chunking.py` | hierarchy, overlap, token budgets, deterministic IDs | no |
| `tests/test_metadata.py` | ID scheme, filter → Cypher translation, concept normalisation | no |
| `tests/test_fusion.py` | RRF arithmetic, dedup, weighting, signal preservation | no |
| `tests/test_trace.py` | trace models, observer events, fusion/rerank/evidence/prompt | no |
| `tests/test_graph_trace.py` | ignored-node reason codes, diagnostic classification | no |
| `tests/test_visualizer_api.py` | FastAPI health, runs, SSE (fake pipeline) | no |
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
│       ├── pipeline.py         HybridRetriever + SocraticRagPipeline
│       ├── trace.py            optional run/stage trace models
│       └── graph_trace.py      bounded ignored-node graph diagnostics
│   └── rag_visualizer/         FastAPI dashboard (SSE, in-memory runs)
├── frontend/                   React + Vite + Cytoscape dashboard
├── scripts/
│   ├── download_curriculum.py
│   ├── replace_corpus.py
│   ├── download_retrieval_models.py
│   ├── neo4j_local.sh
│   ├── init_neo4j.py
│   ├── ingest_corpus.py
│   ├── inspect_graph.py
│   ├── test_retriever.py
│   ├── test_rag.py
│   ├── test_generator.py
│   ├── evaluate_retrieval.py
│   └── run_visualizer.py
├── tests/
├── curriculum/                manifests + alignment committed; raw/ processed git-ignored
├── models/                     git-ignored
│   ├── qwen3-vl-8b-instruct/
│   ├── bge-m3/
│   └── bge-reranker-v2-m3/
├── .neo4j-local/               git-ignored Neo4j tarball + JDK 21 + DB data
├── data/evaluation/          JSONL schema + labelling guide
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
