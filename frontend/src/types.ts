export const STAGE_ORDER = [
  "query",
  "filters",
  "rewrite",
  "image",
  "dense",
  "lexical",
  "graph",
  "fusion",
  "reranker",
  "evidence",
  "prompt",
  "generator",
] as const;

export type StageName = (typeof STAGE_ORDER)[number];

export type StageStatus = "pending" | "running" | "completed" | "skipped" | "failed";

export interface StageTrace {
  name: string;
  status: StageStatus;
  elapsed_ms: number | null;
  summary: string;
  error: string | null;
}

export interface CandidateRow {
  chunk_id: string;
  rank: number | null;
  score: number | null;
  text: string;
  provenance: Record<string, unknown>;
  metadata: Record<string, unknown>;
  entered_fusion: boolean;
  final_evidence: boolean;
}

export interface GraphPath {
  path_id: string;
  seed_chunk_id: string;
  relation: string;
  via: string | null;
  candidate_chunk_id: string;
  accepted: boolean;
  reason_code: string;
  reason_detail: string;
  logical: boolean;
  contribution: number;
  seed_weight: number;
}

export interface GraphNode {
  node_id: string;
  id?: string;
  label: string;
  node_kind: string;
  status: string;
  text: string;
  display_label: string;
  metadata: Record<string, unknown>;
  seed_weight: number | null;
  graph_score: number | null;
  path_ids: string[];
  reason_codes: string[];
  entered_fusion: boolean;
  final_evidence: boolean;
}

export interface GraphEdge {
  edge_id: string;
  source: string;
  target: string;
  relation: string;
  physical: boolean;
  logical: boolean;
  accepted: boolean;
  label: string;
  neo4j_type: string | null;
}

export interface GraphTrace {
  seeds: { chunk_id: string; weight: number }[];
  enabled_relations: string[];
  disabled_relations: { relation: string; reason_code: string | null; detail: string }[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  paths: GraphPath[];
  truncated: boolean;
  truncation_caps: string[];
  truncation_warning: string;
  selected_chunk_ids: string[];
  counters: Record<string, number>;
}

export interface FusionCandidate {
  chunk_id: string;
  dense_rank: number | null;
  dense_contribution: number;
  lexical_rank: number | null;
  lexical_contribution: number;
  graph_rank: number | null;
  graph_contribution: number;
  channels: string[];
  rrf_score: number;
  fused_rank: number;
  text: string;
  provenance: Record<string, unknown>;
}

export interface RerankerCandidate {
  chunk_id: string;
  fused_rank: number | null;
  reranked_rank: number | null;
  rank_movement: number;
  rerank_score: number | null;
  text: string;
  provenance: Record<string, unknown>;
  survived_final_top_k: boolean;
}

export interface EvidenceCheck {
  name: string;
  passed: boolean;
  value: unknown;
  threshold: unknown;
  detail: string;
}

export interface RunTrace {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
  query: string;
  filters: Record<string, unknown>;
  requested_state: string | null;
  retrieval_only: boolean;
  strict: boolean;
  started_at: string;
  completed_at: string | null;
  error: string | null;
  stages: Record<string, StageTrace>;
  dense: {
    model_name: string;
    embedding_dim: number | null;
    query_vector_norm: number | null;
    vector_preview: number[];
    strategy: string;
    used_approximate_index: boolean;
    used_exact_fallback: boolean;
    candidates: CandidateRow[];
  } | null;
  lexical: {
    original_query: string;
    lucene_query: string;
    candidates: CandidateRow[];
  } | null;
  graph: GraphTrace | null;
  fusion: {
    rrf_k: number;
    weight_dense: number;
    weight_fulltext: number;
    weight_graph: number;
    formula: string;
    candidates: FusionCandidate[];
  } | null;
  reranker: { candidates: RerankerCandidate[]; score_kind: string } | null;
  evidence: {
    sufficient: boolean;
    confidence: string;
    checks: EvidenceCheck[];
    kept_chunk_ids: string[];
    reasons: string[];
  } | null;
  prompt: {
    tutor_state: string;
    system_prompt: string;
    user_prompt: string;
    evidence_blocks: { index: number; chunk_id: string; text: string; provenance: Record<string, unknown> }[];
    generation_settings: Record<string, unknown>;
    generation_skipped: boolean;
    skip_reason: string;
  } | null;
  generation: { tokens: string[]; response_text: string; elapsed_ms: number } | null;
  rewrite: {
    original_query: string;
    retrieval_query: string;
    intent: string | null;
    fallback: boolean;
    reason?: string;
    input_kind?: string;
    transcribed_question?: string;
  } | null;
  image: { hits: { image_id: string; score: number; page_number?: number }[]; skipped?: boolean } | null;
  attached_figures: { image_id: string; page_number?: number; score?: number; source?: string }[];
}

export interface QueryFormState {
  query: string;
  grade: string;
  subject: string;
  tutor_state: string;
  retrieval_only: boolean;
  strict: boolean;
  imageFile: File | null;
}

export interface SseEnvelope {
  event: string;
  payload: Record<string, unknown>;
  trace?: RunTrace;
}

export const EMPTY_FORM: QueryFormState = {
  query: "",
  grade: "",
  subject: "",
  tutor_state: "",
  retrieval_only: false,
  strict: false,
  imageFile: null,
};

export function emptyStages(): Record<StageName, StageTrace> {
  const stages = {} as Record<StageName, StageTrace>;
  for (const name of STAGE_ORDER) {
    stages[name] = { name, status: "pending", elapsed_ms: null, summary: "", error: null };
  }
  return stages;
}
