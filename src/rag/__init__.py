"""Hybrid GraphRAG retrieval system for the STEM Socratic Tutor.

Layout (each layer is deliberately independent so it can be swapped or
benchmarked on its own -- no LangChain / LlamaIndex abstraction in between):

    config            centralized, environment-driven settings
    logging_utils     structured logging and stage timers
    schemas           dataclasses for parsed documents, chunks and results
    pdf_parser        PyMuPDF structured extraction (pages, sections, images)
    chunker           hierarchical, token-aware chunking
    concepts          conservative concept extraction and normalisation
    embeddings        BGE-M3 dense embeddings (lazy load, releasable)
    image_embeddings  SigLIP encoder for textbook figures
    image_retriever   image vector kNN + page-chunk fusion
    reranker          BGE-reranker-v2-m3 cross-encoder scoring
    neo4j_store       official Neo4j driver wrapper
    graph_schema      constraints, vector index and full-text index DDL
    ingest            corpus -> graph ingestion with resume support
    dense_retriever   vector-index retrieval channel
    lexical_retriever full-text retrieval channel
    graph_retriever   bounded graph expansion channel
    fusion            reciprocal rank fusion
    evidence          evidence sufficiency gate
    generator         Qwen3-VL-8B-Instruct wrapper with streaming
    query_rewrite     Qwen3-VL-2B-Instruct retrieval query rewriter
    socratic          Socratic prompt/state controller
    pipeline          end-to-end orchestration with full diagnostics

Submodules are imported lazily; importing this package pulls in nothing heavy.
"""

__all__ = [
    "config",
    "logging_utils",
    "schemas",
    "pdf_parser",
    "chunker",
    "concepts",
    "embeddings",
    "reranker",
    "neo4j_store",
    "graph_schema",
    "ingest",
    "dense_retriever",
    "lexical_retriever",
    "graph_retriever",
    "fusion",
    "evidence",
    "query_rewrite",
    "generator",
    "socratic",
    "pipeline",
]
