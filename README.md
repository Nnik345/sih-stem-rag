# SIH STEM RAG

Local research project for collecting and processing Core Knowledge Foundation STEM curriculum materials (Grades 1–3 Mathematics and Science).

## Python version

**Python 3.12.14**

Create a Python environment with 3.12.14, then from the repository root:

```bash
pip install -r requirements.txt
```

The download script currently needs no third-party packages; `requirements.txt` is included so dependencies can be added as the project grows.

## Downloading the dataset

Curriculum PDFs are downloaded from the official Core Knowledge Foundation site:

https://www.coreknowledge.org/download-free-curriculum/

The script collects all freely available PDF resources for:

- Grade 1 Mathematics and Science
- Grade 2 Mathematics and Science
- Grade 3 Mathematics and Science

### Run the downloader

```bash
python scripts/download_core_knowledge_stem.py
```

Optional: save a log of the run:

```bash
python scripts/download_core_knowledge_stem.py 2>&1 | tee download.log
```

### What it does

1. Crawls the official curriculum listing for Grades 1–3 Mathematics and Science.
2. Visits each unit page and downloads every resource with a **Download Free PDF Version** link.
3. Saves files under `core_knowledge_stem/` using this layout:

```
core_knowledge_stem/
├── grade_01/
│   ├── mathematics/
│   └── science/
├── grade_02/
│   ├── mathematics/
│   └── science/
├── grade_03/
│   ├── mathematics/
│   └── science/
└── manifest.json
```

Each unit folder contains `student/`, `teacher/`, and `other/` subdirectories based on resource type.

4. Writes `core_knowledge_stem/manifest.json` with metadata for every downloaded PDF.
5. Re-crawls the site and verifies that all expected units and PDFs are present locally.

### Re-running

The script skips PDFs that already exist locally, so it is safe to re-run if a download was interrupted.

### Expected output

A full run downloads **51 units** and **126 PDFs** (~935 MB). On completion, the script prints a summary with unit counts, PDF counts, total size, and any failures.

### Important notes

- Materials are for **local research use only** and must not be redistributed.
- Only official Core Knowledge Foundation URLs are used; no login or paid resources are accessed.
- Downloaded data is excluded from git via `.gitignore` and should not be committed to GitHub.

## Project structure

```
sih-stem-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   └── download_core_knowledge_stem.py
└── core_knowledge_stem/
    └── manifest.json
```
