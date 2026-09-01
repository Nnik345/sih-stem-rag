# SIH STEM RAG

Local research project building a **hallucination-resistant Socratic STEM tutor**
over official English-medium **NCERT** Mathematics and Science textbooks for
**CBSE classes 1–12** (EVS in the primary years; Physics, Chemistry and Biology
in classes 11–12). NCERT books *are* the CBSE curriculum here: they are both
the ingested corpus and the alignment authority. There is no CISCE layer and
no separate CBSE syllabus PDF.

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

Download the official English NCERT STEM textbooks from ncert.nic.in. See
[Dataset](#dataset).

```bash
python scripts/download_curriculum.py
```

### 4. Models

```bash
python scripts/download_retrieval_models.py
python scripts/download_qwen_models.py
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
`what are the components of food` with grade `6` and
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
    --query "what are the components of food" \
    --grade 6 --subject science

python scripts/test_rag.py \
    --query "how does light reflect from a plane mirror" \
    --grade 10 --subject science

python scripts/test_rag.py \
    --query "what is electrostatics" \
    --grade 12 --subject physics --retrieval-only
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

Fetch official English NCERT STEM textbooks from
[ncert.nic.in/textbook.php](https://ncert.nic.in/textbook.php). Only catalog
URLs are used; there is no login and no unofficial mirror.

```bash
python scripts/download_curriculum.py
```

Output: `curriculum/raw/ncert/` (git-ignored). Details in [Dataset](#dataset).

### 4. Retrieval models (BGE + 2B rewriter)

```bash
python scripts/download_retrieval_models.py
```

Output: `models/bge-m3/` (~2.3 GB), `models/bge-reranker-v2-m3/` (~2.3 GB),
`models/siglip-base-patch16-224/` (figure matching), and
`models/qwen3-vl-2b-instruct/` (query rewriter, ~4 GB). If a model is already
complete locally, the download script skips it and prints a status table. Use
`--force` to re-download. Details in [Models](#models).

### 5. Qwen models (2B rewriter + 8B tutor)

```bash
python scripts/download_qwen_models.py
```

Output: `models/qwen3-vl-2b-instruct/` and `models/qwen3-vl-8b-instruct/`
(git-ignored). The 2B rewriter is also fetched by `download_retrieval_models.py`;
already-complete directories are skipped. The 8B tutor is **not** fetched by the
retrieval script.

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
python scripts/embed_images.py     # SigLIP over existing Image nodes
```

Ingestion writes git-ignored artefacts under `data/processed/` (images,
manifests, cache) and populates Neo4j. Re-running is idempotent unless you pass
`--force`.

### 8. Smoke verification

```bash
python scripts/test_retriever.py \
    --query "what are the components of food" \
    --grade 6 --subject science

python scripts/test_rag.py \
    --query "how does light reflect from a plane mirror" \
    --grade 10 --subject science

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
| `curriculum/raw/` | `python scripts/download_curriculum.py` (official NCERT zips) | several hundred MB |
| `models/bge-m3/` | `python scripts/download_retrieval_models.py` | ~2.3 GB |
| `models/bge-reranker-v2-m3/` | same script | ~2.3 GB |
| `models/qwen3-vl-2b-instruct/` | `python scripts/download_retrieval_models.py` or `python scripts/download_qwen_models.py` | ~4 GB |
| `models/siglip-base-patch16-224/` | `python scripts/download_retrieval_models.py --model siglip-base-patch16-224` | ~400 MB |
| `models/qwen3-vl-8b-instruct/` | `python scripts/download_qwen_models.py` | ~17 GB |
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
| `.ppt-assets/` | local PowerPoint working diagrams; not required to run the tutor | — |

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
   Metadata filtering          (grade + subject, required from the caller)
        |
   Qwen3-VL-2B query rewrite   (text and optional student photo; then unloaded)
        |
   Mathematics query specialisation (d/dx / derivative cues, when subject is maths)
        |
   SigLIP image kNN            (if a photo was uploaded; then unloaded)
        |
   Retrieval channels
        |-- dense semantic retrieval
        |-- lexical / full-text retrieval
        |-- bounded graph expansion
        +-- page chunks from matched textbook figures
        |
   Weighted Reciprocal Rank Fusion
        |
   BGE-reranker-v2-m3
        |
   Grade-aware final evidence
        |
   Evidence sufficiency gate
        |
   Qwen3-VL-8B-Instruct        (buffered structured JSON, then Python format)
        |
   Validated Socratic tutoring response
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

Supported grades are **1–12**. The corpus is official English NCERT STEM
textbooks that implement the **CBSE** curriculum. NCERT is both the ingested
source and the alignment authority (`source_id = ncert_textbook`). There is no
separate CBSE syllabus PDF and no CISCE layer.

Physics, Chemistry and Biology at classes 11–12 ingest as those subjects
(`physics`, `chemistry`, `biology`), not as a single `science` bucket. Classes
3–10 science (including primary EVS) stay `science`.

| Classes | Mathematics | Science |
| ------: | ----------- | ------- |
| 1–2 | Joyful Mathematics | — |
| 3–5 | Maths Mela | Our Wondrous World (EVS) as `science` |
| 6 | Ganita Prakash | Curiosity |
| 7–8 | Ganita Prakash (parts) | Curiosity |
| 9 | Ganita Manjari | Exploration |
| 10 | Mathematics | Science |
| 11–12 | Mathematics (parts) | Physics, Chemistry, Biology as separate subjects |

Hindi/Urdu editions, Exemplar Problems, Lab Manuals, covers, and **Answers**
PDFs are not ingested. Unsolved `Exercises` / `Let’s practise` sections are
`practice_only`; answer keys are `evaluation_only`.

### Run the downloader

```bash
python scripts/download_curriculum.py
```

Only URLs in `src/rag/curriculum_catalog.py` on `ncert.nic.in` are used. There
is no login and no mirror fallback. Valid chapter directories are skipped on
re-run. SHA-256 hashes are written to `curriculum/manifests/sources.yaml` and
`curriculum/manifests/checksums.sha256`.

Each book is the official complete-book zip
(`https://ncert.nic.in/textbook/pdf/{code}dd.zip`). Chapter PDFs extract to:

```
curriculum/raw/ncert/{subject}/grade_{nn}/{book_slug}/student/
```

Ingest creates **one Document per chapter PDF**.

Official page: https://ncert.nic.in/textbook.php

Do not use Internet Archive, Vedantu, or other unofficial mirrors.

Layout:

```
curriculum/
├── manifests/sources.yaml
├── raw/ncert/    git-ignored downloads
└── processed/    git-ignored ingest artefacts
```

### Replace the old graph, then ingest

```bash
python scripts/replace_corpus.py --purge-all-curriculum --yes
python scripts/download_curriculum.py
python scripts/ingest_corpus.py
```

Ordinary ingestion does not delete the graph. A full NCERT ingest is much
larger than the old 25-document corpus and may take hours.

### Important notes

- Materials are for **local research use only**.
- NCERT textbooks are copyrighted; use the official ncert.nic.in files only.
- `--strict` / `alignment_strict` keeps retrieval on `source_id = ncert_textbook`.

## Models

All five models are stored locally and none are committed to the repository.

| Role | Model | Local path |
|------|-------|------------|
| Query rewriter | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | `models/qwen3-vl-2b-instruct` |
| Generator | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (original, non-quantized) | `models/qwen3-vl-8b-instruct` |
| Embeddings | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | `models/bge-m3` |
| Reranker | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `models/bge-reranker-v2-m3` |
| Figure matching | [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) | `models/siglip-base-patch16-224` |

### Query rewriter

Qwen3-VL-2B-Instruct rewrites the student's wording into an NCERT-like retrieval
query, then is unloaded before SigLIP (if a photo was uploaded) and BGE-M3.
With a photo it also classifies `input_kind` (`math_problem`, `diagram`, or
`other`) and may transcribe the problem for the tutor. It does not answer the
question and does not infer grade or subject. Check-my-work questions (for example “is the
differentiation of x² + 3x = 2x + 3?”) are labelled `verify`: retrieval targets
the topic / curriculum rule, not the student’s proposed answer or their specific
polynomial, and does not treat both sides as things to differentiate.
After rewrite, mathematics queries are specialised in
`specialize_maths_retrieval_query`: cues such as `d/dx`, `dy/dx`, `derivative`,
`differentiate`, and `differentiation` (including on the original question or
photo transcript) are normalised to NCERT-like differentiation-rule search
terms rather than quadratic roots or polynomial zeroes. Cues on the student’s
original wording override a rewriter that drifted into “quadratic equations and
roots”. For example:

```text
provide a solution for d/dx 3x^2 - 4x + 3
```

Retrieval searches for the applicable derivative rules, not a request to solve
a quadratic equation. The specialised search
string is used for dense, lexical, and rerank retrieval. The evidence-gate
overlap check uses the student’s wording (or the photo transcript), not that
search string. The original question stays in the Socratic user prompt. Intent
is diagnostic only and does not pick the tutoring state.

Mathematics and science/PCB retrieval treat the selected class as **where the
student is now**. Index scans keep the requested class and earlier classes in
the same subject lineage (`grade <=` the requested class; physics/chemistry/biology
may use class 3–10 science). Material above the requested class is not
retrieved. After reranking, final evidence prefers the requested class, then
closer earlier classes when relevance is comparable; topic relevance still
outranks grade proximity (see [Grade-aware final evidence](#grade-aware-final-evidence)).
The other lineage (maths ↮ science) stays closed.

### Generator

Qwen3-VL-8B-Instruct is loaded with Hugging Face Transformers using
`device_map="auto"` and a **hardware-aware** weight map. The loader reads free
VRAM and physical RAM at load time: generation headroom stays on the GPU
(vision prefill + KV cache), leftover layers go to system RAM (up to
`GENERATOR_CPU_RAM_FRACTION` of RAM, default 80%). A 32 GiB card can run the 8B
tutor fully on GPU. A 12 GiB or 8 GiB card offloads more layers and generation
is slower. Tutoring replies are generated with buffered, non-streamed
`generator.complete()` (`do_sample=False`), then parsed and formatted in
Python before `scripts/test_rag.py` or the dashboard shows the student text.

The dashboard releases BGE / reranker / SigLIP before the tutor loads. On small
GPUs the tutor is also unloaded before the next retrieval so the 2B rewriter
fits; a 32 GiB card can keep the tutor resident.

Transformers is installed from GitHub because Qwen3-VL support was not in a
tagged release at setup time. **Do not replace that dependency.**

Download both Qwen checkpoints once:

```bash
python scripts/download_qwen_models.py
```

Download the retrieval models (BGE-M3, BGE reranker, and the 2B rewriter) with:

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

The four models are not all loaded at once. Each is loaded lazily and can be
released:

```
ingest: load BGE-M3 -> embed -> release BGE-M3
query : load Qwen3-VL-2B rewriter -> rewrite -> release rewriter
        load BGE-M3 -> dense retrieval -> (reranker) -> release embedder/reranker
        load Qwen3-VL-8B (weights split from live VRAM + RAM) -> buffered structured reply
```

No quantization is used anywhere. On a small GPU the 8B tutor is unloaded before
the next query's rewriter so both fit; a 32 GiB card can keep it loaded.

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
unit_id     grade_06:science:curiosity
document_id <unit_id>:student:fecu101
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
retriever.retrieve("what are the components of food", grade=6, subject="science")
```

`grade` and `subject` are required on the dashboard, API and CLI. They are never
inferred from the question text: the application already knows the student's
class, and guessing it would be both unreliable and pedagogically wrong.
`HybridRetriever.retrieve` still accepts optional filters for isolated tests.
Unit, resource type, audience and document id are not user fields; retrieved
chunk provenance (unit title, pages) is still passed to the generator in the
evidence block.

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
BGE-reranker-v2-m3. Reranker logits are kept on every candidate; the final
evidence list is not “top 5 rerank scores” alone.

### 7. Grade-aware final evidence

`select_final_evidence` in `src/rag/fusion.py` then chooses up to
`final_top_k` (default 5) chunks. Retrieval already restricts the index scan to
the requested class and earlier classes (`grade <=` requested). Final selection
then:

- Gives the **requested grade** the highest priority among comparably relevant
  chunks.
- Keeps **earlier grades** available for prerequisite material.
- Prefers a **closer earlier grade** over a more distant one when relevance is
  comparable. For Class 12 that order is Class 12, then Class 11, then Class 10,
  and so on.
- Treats **topic relevance as more important than grade proximity**. An
  off-topic chunk from the requested class does not displace a clearly more
  relevant earlier-class passage.
- **Drops material above the student’s requested grade.**

Diagnostics record `grade proximity (requested class …; closer grades preferred
among comparable relevance)`.

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
comes from the requested grade and subject (or a high-scoring earlier class in
the same subject lineage); and whether the retrieved text overlaps the
question's content terms. For mathematics, overlap ignores algebraic instance
tokens such as `3x` or `plus` so a power-rule passage can support
`x² + 3x`.

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
`GIVE_HINT` (default), `EXPLAIN_CONCEPT`, `CONFIRM_ANSWER`, and
`INSUFFICIENT_EVIDENCE`. Student state is not modelled across turns, but the
interface takes a conversation history so it does not assume one-shot answers.

The generator is instructed to treat retrieved curriculum as the factual basis
and not to fabricate curriculum facts. Source metadata (document, page, unit,
chunk) is preserved internally so citations can be surfaced later; no URLs are
fabricated. The student never sees evidence labels such as `[E1]`.

### Tutoring states

Word limits are enforced on the formatted student-facing reply
(`src/rag/socratic.py`, `_STATE_WORD_LIMITS`):

| State | Word limit | Behaviour |
|-------|------------|-----------|
| `GIVE_HINT` | 90 | One small hint toward the next useful rule or step, then exactly one guiding question. Does not reveal the answer or give a full solution. |
| `EXPLAIN_CONCEPT` | 450 | Explains the concept from the evidence. For mathematics, states the relevant formula or rule when one applies, then a fully worked example that is similar but **not** the student’s exact problem (different values or expression). Does not solve the student’s instance. For science, a formula or example is optional when the evidence supports it. |
| `CONFIRM_ANSWER` | 300 | Checks the student’s attempt. Algebraically or scientifically equivalent answers count as correct. If correct: a brief confirmation prefixed `Correct.` If incorrect: `Not quite.`, then mistakes grouped with a corrective hint each, without revealing the corrected final answer or a replacement solution. |
| `INSUFFICIENT_EVIDENCE` | 130 | Chosen automatically when the evidence gate fails (not a `--state` value). Declines: verified curriculum evidence is insufficient. Does not answer from general knowledge. |

### Buffered structured generation

Every tutoring state uses the same generation path
(`src/rag/structured_tutor.py`, `src/rag/tutor_json.py`; `CONFIRM_ANSWER` via
`src/rag/confirm_eval.py`):

- Buffered, non-streamed `generator.complete()` (not token streaming).
- Deterministic settings: `do_sample=False` and a per-state token cap.
- State-specific structured JSON from the model.
- Parse and validate that JSON before any reply reaches the student.
- Exactly one JSON-repair retry after invalid output.
- A safe, state-specific fallback if generation still fails.
- Deterministic Python formatting of the validated fields into student text.

The student is not shown raw evaluator JSON, private checking text, provisional
verdicts, or evidence labels. Structural validation checks shape, leaks, word
limits, and similar constraints; **it does not guarantee that every scientific
or mathematical claim is factually correct.**

### Mathematical verification

For mathematics `CONFIRM_ANSWER` turns, a question that is safely parseable as
“is the derivative/differentiation of EXPR equal to RESULT?” may be checked with
SymPy (`sympy` in `requirements.txt`) before the reply is formatted.

- Algebraically equivalent expressions are accepted.
- Omitted zero terms and reordered equivalent terms are accepted.
- Unsupported, unsafe, or multi-variable expressions are not forced through
  SymPy; the tutor falls back to evidence-grounded model evaluation.
- SymPy is not used for non-mathematical subjects.

## Commands

Run all commands from the repository root.

### 1. Download curriculum sources

```bash
python scripts/download_curriculum.py
```

Fetches official English NCERT STEM complete-book zips from ncert.nic.in and
extracts chapter PDFs. Existing valid chapter directories are skipped. See
[Dataset](#dataset).

### 2. Download the retrieval models

```bash
python scripts/download_retrieval_models.py
python scripts/download_retrieval_models.py --model bge-m3
python scripts/download_retrieval_models.py --model qwen3-vl-2b-instruct
python scripts/download_retrieval_models.py --model siglip-base-patch16-224
python scripts/download_retrieval_models.py --force
```

Downloads BGE-M3, BGE-reranker-v2-m3, SigLIP, and Qwen3-VL-2B-Instruct only if missing.
Already-complete models are skipped and reported in a status table at the end.
Never touches the 8B tutor (use `scripts/download_qwen_models.py` for 8B + 2B).
Flags: `--model` (`bge-m3`, `bge-reranker-v2-m3`, `siglip-base-patch16-224`, or
`qwen3-vl-2b-instruct`, repeatable), `--force` (re-download even when complete).

### 3. Download the Qwen models

```bash
python scripts/download_qwen_models.py
python scripts/download_qwen_models.py --model qwen3-vl-2b-instruct
```

### 4. Start Neo4j

Requires a provisioned `.neo4j-local/` (see [Neo4j setup](#neo4j-setup)).

```bash
./scripts/neo4j_local.sh start
```

### 5. Initialise the schema

```bash
python scripts/init_neo4j.py
```

Creates constraints, property indexes, the chunk vector index (dimension read from
BGE-M3), the image vector index when SigLIP is present, and the full-text indexes.
Idempotent. Flags: `--show` (report schema and counts only), `--reset` (delete all
nodes; prompts unless `--yes`), `--drop-indexes` (with `--reset`, also drop indexes).

### 6. Ingest the corpus

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
python scripts/ingest_corpus.py --parse-workers 8             # parallel PDF parse
python scripts/ingest_corpus.py --reset --yes                 # skip prompt
```

Embed textbook `Image` nodes with SigLIP (does not re-parse PDFs). Skip if
`embedding_version` already matches. Student photos are never written here.

```bash
python scripts/embed_images.py
python scripts/embed_images.py --force
```

### 7. Inspect the graph

```bash
python scripts/inspect_graph.py
python scripts/inspect_graph.py --units
python scripts/inspect_graph.py --unit grade_06:science:curiosity
python scripts/inspect_graph.py --chunk "<chunk_id>"
python scripts/inspect_graph.py --concepts 30
python scripts/inspect_graph.py --images 10
```

### 8. Test retrieval (no generator)

```bash
python scripts/test_retriever.py --query "what are the components of food" --grade 6 --subject science
```

Prints dense, full-text, graph, fused and reranked results side by side with per
channel ranks, scores, RRF scores and reranker scores. This script never calls
the generator. Useful flags: `-q`/`--query`, `-g`/`--grade`, `-s`/`--subject`,
`--limit`, `--final-top-k`, `--no-rerank`, `--images`, `--json`. Grade and
subject are required.

### 9. Test the full RAG pipeline

```bash
python scripts/test_rag.py --query "what are the components of food" --grade 6 --subject science
python scripts/test_rag.py --query "what are the components of food" --grade 6 --subject science --state GIVE_HINT
python scripts/test_rag.py --query "how does light reflect from a plane mirror" --grade 10 --subject science --strict
python scripts/test_rag.py --query "what is electrostatics" --grade 12 --subject physics --retrieval-only
python scripts/test_rag.py --image path/to/photo.jpg --grade 9 --subject science
```

Runs filtering, rewrite (with optional photo classify/transcribe), image kNN when
a photo is given, fusion, reranking, grade-aware evidence selection and the
evidence gate, prints concise diagnostics, then loads Qwen3-VL-8B and prints the
validated Socratic reply (buffered structured generation, not token-by-token
streaming). Textbook figures are not shown in the
answer. If the gate reports insufficient
evidence, the generator is either asked to decline (`--strict` off) or not loaded
at all (`--strict` on).

Omit `-q`/`--query` (and optionally `-g`/`-s`) to be prompted interactively.
`--image` may be used without a typed question. Grade and subject are always
required.

#### CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-q`, `--query` | *(prompt)* | Student question. Optional when `--image` is set |
| `--image PATH` | — | Student photo (not ingested). Query may be omitted |
| `-g`, `--grade` | *(required)* | Class `1`–`12` |
| `-s`, `--subject` | *(required)* | Gated by class: 1–2 maths; 3–10 maths/science; 11–12 maths/physics/chemistry/biology |
| `--state` | `GIVE_HINT` | Tutoring move (see table below) |
| `--retrieval-only` | off | Stop after the evidence gate; never load the generator |
| `--strict` | off | On insufficient evidence, skip generation entirely |
| `--max-new-tokens` | from config (`640`) | Writes `GENERATOR_MAX_NEW_TOKENS` on the config object. Tutoring uses buffered `do_sample=False` generation with per-state token caps in `src/rag/tutor_json.py` |
| `--json PATH` | — | Write the full result record (diagnostics, gate, response) to a file |
| `--log-level` | `WARNING` | Logging verbosity |

#### `--state` tutoring moves

These change how the generator is prompted. They do **not** bypass the evidence
gate. Leave `--state` blank to use `GIVE_HINT`. Word limits and JSON shapes are
described under [Socratic controller](#socratic-controller).

| Value | Behaviour |
|-------|-----------|
| `GIVE_HINT` | Default. One small hint and one guiding question; do not reveal the answer (90 words). |
| `EXPLAIN_CONCEPT` | Explain the concept. Maths: formula or rule plus a similar worked example, not the student’s exact problem (450 words). |
| `CONFIRM_ANSWER` | The question may include the problem and the student’s attempt. Accept equivalents; confirm briefly if correct, or group mistakes with hints if not (300 words). |

`INSUFFICIENT_EVIDENCE` is chosen automatically when the gate fails — it is not a
valid `--state` value. With `--strict`, generation is skipped instead of running
in that state.

### 10. Generator sanity check (pre-existing)

```bash
python scripts/test_generator.py
```

### 11. Evaluate retrieval (needs manual labels)

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
| `REWRITER_MODEL_PATH` | `models/qwen3-vl-2b-instruct` | Qwen3-VL-2B query rewriter |
| `GENERATOR_MODEL_PATH` | `models/qwen3-vl-8b-instruct` | Qwen3-VL-8B tutor |
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
| `EVIDENCE_REQUIRE_SCOPE_MATCH` | `true` | Evidence must come from this class or a high-scoring earlier class in the same subject lineage |
| `EVIDENCE_MIN_QUERY_TERM_OVERLAP` | `0.15` | Fraction of query content words that must appear |
| `EVIDENCE_MIN_PRIOR_GRADE_RERANK_SCORE` | `1.0` | Rerank floor for earlier-class chunks (stricter than `EVIDENCE_MIN_RERANK_SCORE`) |
| `IMAGE_EMBEDDING_MODEL_PATH` | `models/siglip-base-patch16-224` | SigLIP figure encoder |
| `IMAGE_TOP_K` | `8` | Image-vector neighbours |
| `IMAGE_MIN_SCORE` | `0.25` | Cosine floor for preferred output figures |
| `WEIGHT_IMAGE` | `1.0` | Image-channel RRF weight |
| `MULTIMODAL_ENABLED` | `true` | Student photo input (rewrite, image kNN, tutor vision). Textbook figures are not shown in the answer |
| `GENERATOR_VRAM_RESERVE_GIB` | `5.5` | Floor GiB held back from 8B weights for vision prefill / KV cache (also scales with the card) |
| `GENERATOR_VRAM_HEADROOM_FRACTION` | `0.25` | Extra headroom as a fraction of total VRAM (capped at 10 GiB) |
| `GENERATOR_CPU_RAM_FRACTION` | `0.80` | Fraction of physical RAM for CPU-offloaded 8B layers |
| `GENERATOR_MAX_IMAGE_PIXELS` | `0` | `0` = pick Qwen-VL `max_pixels` from VRAM size |
| `GENERATOR_MAX_EVIDENCE_CHARS` | `1000` | Per-chunk evidence character budget |
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
| `tests/test_fusion.py` | RRF arithmetic, dedup, weighting, grade-proximity final evidence | no |
| `tests/test_query_rewrite.py` | rewrite JSON, maths specialisation (`d/dx`), figure-need cues | no |
| `tests/test_socratic.py` | tutoring states, word limits, prompts | no |
| `tests/test_structured_tutor.py` | buffered JSON path for all tutor states | no |
| `tests/test_confirm_eval.py` | CONFIRM_ANSWER formatting, SymPy equivalents, no token streaming | no |
| `tests/test_evidence.py` | gate checks, maths instance overlap | no |
| `tests/test_trace.py` | trace models, observer events, fusion/rerank/evidence/prompt | no |
| `tests/test_graph_trace.py` | ignored-node reason codes, diagnostic classification | no |
| `tests/test_visualizer_api.py` | FastAPI health, runs, SSE (fake pipeline) | no |
| `tests/test_multimodal.py` | rewrite image JSON, path sandbox, lineage, skip-live embed/kNN | no (live parts skip) |
| `tests/test_model_memory.py` | GPU/RAM weight split, pixel cap, release-before-generate | no |
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
│       ├── image_embeddings.py SigLIP textbook-figure encoder
│       ├── image_retriever.py  image kNN + page-chunk fusion
│       ├── image_index.py      embed existing Image nodes
│       ├── image_paths.py      curriculum-path sandbox
│       ├── image_serve.py      browser-safe PNG conversion for figures
│       ├── reranker.py         BGE-reranker-v2-m3
│       ├── neo4j_store.py      driver wrapper, error translation
│       ├── graph_schema.py     labels, constraints, vector/full-text indexes
│       ├── ingest.py           idempotent ingestion
│       ├── retrieval_base.py   shared projection + filter builder
│       ├── dense_retriever.py  vector channel
│       ├── lexical_retriever.py full-text channel
│       ├── graph_retriever.py  bounded graph expansion
│       ├── fusion.py           weighted RRF + grade-aware final evidence
│       ├── evidence.py         evidence sufficiency gate
│       ├── query_rewrite.py    Qwen3-VL-2B retrieval query rewriter
│       ├── socratic.py         tutoring controller and prompts
│       ├── tutor_json.py       buffered JSON parse, repair, leak checks
│       ├── structured_tutor.py structured generation for all tutor states
│       ├── confirm_eval.py     CONFIRM_ANSWER JSON + optional SymPy check
│       ├── generator.py        Qwen3-VL-8B wrapper (stream helper + buffered complete)
│       ├── model_memory.py     adaptive GPU/RAM weight placement
│       ├── pipeline.py         HybridRetriever + SocraticRagPipeline
│       ├── trace.py            optional run/stage trace models
│       └── graph_trace.py      bounded ignored-node graph diagnostics
│   └── rag_visualizer/         FastAPI dashboard (SSE, in-memory runs)
├── frontend/                   React + Vite + Cytoscape dashboard
├── scripts/
│   ├── download_curriculum.py
│   ├── replace_corpus.py
│   ├── download_retrieval_models.py
│   ├── download_qwen_models.py
│   ├── neo4j_local.sh
│   ├── init_neo4j.py
│   ├── ingest_corpus.py
│   ├── embed_images.py
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
│   ├── qwen3-vl-2b-instruct/
│   ├── bge-m3/
│   ├── bge-reranker-v2-m3/
│   └── siglip-base-patch16-224/
├── .neo4j-local/               git-ignored Neo4j tarball + JDK 21 + DB data
├── data/evaluation/          JSONL schema + labelling guide
```

There is no `docker-compose.yml`. Neo4j is started with `scripts/neo4j_local.sh`
after provisioning `.neo4j-local/` as described in [Neo4j setup](#neo4j-setup).

## Current limitations

- **BGE-M3 sparse (lexical-weight) retrieval is not implemented.** Lexical
  matching comes from the Neo4j full-text index instead.
- **ColBERT / multi-vector late-interaction retrieval is not implemented.**
- **Student photos are never stored as `Image` nodes.** Uploads are temporary
  query files, deleted after the run. Output figures are only Neo4j `Image`
  rows (files under `curriculum/processed/images/`). Textbook figures are not
  shown in the answer. A student photo is input only (rewrite, retrieval, tutor).
  The 8B tutor must not draw or invent a diagram.
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
- Generation can be slow, especially when layers are CPU-offloaded on 8–12 GiB GPUs.

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
