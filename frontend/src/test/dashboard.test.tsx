import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EvidencePanel } from "../components/EvidencePanel";
import { FusionPanel } from "../components/FusionPanel";
import { GraphPanel } from "../components/GraphPanel";
import { NodeDrawer } from "../components/NodeDrawer";
import { PipelineRail } from "../components/PipelineRail";
import { LiveResponse } from "../components/LiveResponse";
import { QueryPanel } from "../components/QueryPanel";
import { RerankerPanel } from "../components/RerankerPanel";
import {
  edgeVisible,
  nodeVisible,
  visualRoleForNode,
  type GraphFilterState,
} from "../graphStyles";
import { applyTraceStages, parseSseBlock } from "../sse";
import type { GraphEdge, GraphNode, RunTrace } from "../types";
import { EMPTY_FORM, emptyStages } from "../types";
import { validateQueryForm } from "../validation";

vi.mock("cytoscape", () => {
  const cy = {
    on: vi.fn(),
    elements: () => ({ remove: vi.fn() }),
    add: vi.fn(),
    layout: () => ({ run: vi.fn() }),
    getElementById: () => ({ length: 0, select: vi.fn(), data: vi.fn() }),
    nodes: () => ({ unselect: vi.fn() }),
    fit: vi.fn(),
    destroy: vi.fn(),
  };
  const factory = () => cy;
  factory.use = vi.fn();
  return { default: factory };
});
vi.mock("cytoscape-dagre", () => ({ default: {} }));
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: unknown }) => children,
  BarChart: ({ children }: { children: unknown }) => <div>{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

function baseTrace(overrides: Partial<RunTrace> = {}): RunTrace {
  return {
    run_id: "abc",
    status: "running",
    query: "how does weather change",
    filters: { grade: 3, subject: "science" },
    requested_state: null,
    retrieval_only: false,
    strict: false,
    started_at: "now",
    completed_at: null,
    error: null,
    stages: emptyStages(),
    dense: null,
    lexical: null,
    graph: null,
    fusion: null,
    reranker: null,
    evidence: null,
    prompt: null,
    generation: null,
    rewrite: null,
    image: null,
    attached_figures: [],
    ...overrides,
  };
}

function node(partial: Partial<GraphNode> & Pick<GraphNode, "node_id">): GraphNode {
  return {
    label: "Chunk",
    node_kind: "candidate",
    status: "ignored",
    text: "",
    display_label: partial.node_id,
    metadata: {},
    seed_weight: null,
    graph_score: null,
    path_ids: [],
    reason_codes: [],
    entered_fusion: false,
    final_evidence: false,
    ...partial,
  };
}

describe("query validation", () => {
  it("rejects a grade outside 1-12", () => {
    expect(validateQueryForm({ ...EMPTY_FORM, query: "food", grade: "13" }).grade).toMatch(/1 through 12/i);
    expect(validateQueryForm({ ...EMPTY_FORM, query: "food", grade: "6", subject: "science" }).grade).toBeUndefined();
  });

    it("requires grade and subject and hides extra metadata fields", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <QueryPanel
        form={EMPTY_FORM}
        errors={{}}
        running={false}
        onChange={() => undefined}
        onSubmit={onSubmit}
      />,
    );
    await user.click(screen.getByRole("button", { name: /run pipeline/i }));
    expect(screen.getByText(/enter a student question/i)).toBeInTheDocument();
    expect(screen.getByText(/grade is required/i)).toBeInTheDocument();
    expect(screen.getByText(/subject is required/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/unit id/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/resource type/i)).toBeNull();
    expect(screen.queryByLabelText(/audience/i)).toBeNull();
    expect(screen.queryByLabelText(/document id/i)).toBeNull();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("gates subjects by grade", () => {
    expect(validateQueryForm({ ...EMPTY_FORM, query: "q", grade: "6", subject: "physics" }).subject).toMatch(
      /not offered/i,
    );
    expect(validateQueryForm({ ...EMPTY_FORM, query: "q", grade: "12", subject: "physics" }).subject).toBeUndefined();
  });

  it("allows an empty question when an image is attached", () => {
    const file = new File(["x"], "cell.png", { type: "image/png" });
    expect(
      validateQueryForm({ ...EMPTY_FORM, grade: "6", subject: "science", imageFile: file }).query,
    ).toBeUndefined();
  });

  it("offers hint, explain, and confirm-answer states", () => {
    render(
      <QueryPanel
        form={EMPTY_FORM}
        errors={{}}
        running={false}
        onChange={() => undefined}
        onSubmit={() => undefined}
      />,
    );
    const select = screen.getByLabelText(/tutor state/i);
    const labels = Array.from(select.querySelectorAll("option")).map((option) => option.textContent);
    expect(labels).toEqual([
      "GIVE_HINT (default)",
      "GIVE_HINT",
      "EXPLAIN_CONCEPT",
      "CONFIRM_ANSWER",
    ]);
    expect(labels.join(" ")).not.toMatch(/ASK_QUESTION|CORRECT_MISCONCEPTION|CONFIRM_STEP/);
  });
});

describe("pipeline rail", () => {
  it("updates stage states", async () => {
    const stages = emptyStages();
    stages.dense = { name: "dense", status: "running", elapsed_ms: null, summary: "searching", error: null };
    const onSelect = vi.fn();
    render(<PipelineRail stages={stages} active="query" onSelect={onSelect} />);
    expect(screen.getByText("running")).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: /dense/i }));
    expect(onSelect).toHaveBeenCalledWith("dense");
  });
});

describe("SSE helpers", () => {
  it("parses named events and applies stage snapshots", () => {
    const parsed = parseSseBlock('event: dense_completed\ndata: {"event":"dense_completed","trace":{"stages":{"dense":{"name":"dense","status":"completed","elapsed_ms":12,"summary":"3","error":null}}}}');
    expect(parsed?.event).toBe("dense_completed");
    const trace = parsed?.data?.trace as RunTrace;
    const stages = applyTraceStages(trace);
    expect(stages.dense.status).toBe("completed");
    expect(stages.dense.elapsed_ms).toBe(12);
  });
});

describe("graph styles and filters", () => {
  const filters: GraphFilterState = {
    showContext: true,
    showAccepted: true,
    showIgnored: true,
    relations: new Set(),
    search: "",
  };

  it("maps statuses to visual roles", () => {
    expect(visualRoleForNode(node({ node_id: "s", node_kind: "seed", status: "seed" }))).toBe("seed");
    expect(
      visualRoleForNode(node({ node_id: "s-ev", node_kind: "seed", status: "seed", final_evidence: true })),
    ).toBe("evidence");
    expect(visualRoleForNode(node({ node_id: "a", status: "accepted" }))).toBe("accepted");
    expect(visualRoleForNode(node({ node_id: "e", final_evidence: true }))).toBe("evidence");
    expect(
      visualRoleForNode(node({ node_id: "m", reason_codes: ["METADATA_FILTER_MISMATCH"] })),
    ).toBe("metadata_mismatch");
    expect(visualRoleForNode(node({ node_id: "h", node_kind: "hub", reason_codes: ["HUB_CONCEPT_EXCLUDED"] }))).toBe("hub");
    expect(visualRoleForNode(node({ node_id: "c", node_kind: "context", status: "context" }))).toBe("context");
    expect(visualRoleForNode(node({ node_id: "i" }))).toBe("ignored");
  });

  it("toggles ignored and context nodes", () => {
    const ignored = node({ node_id: "i" });
    const context = node({ node_id: "ctx", node_kind: "context", status: "context", label: "Section" });
    expect(nodeVisible(ignored, { ...filters, showIgnored: false })).toBe(false);
    expect(nodeVisible(context, { ...filters, showContext: false })).toBe(false);
    expect(nodeVisible(ignored, filters)).toBe(true);
    expect(nodeVisible(context, filters)).toBe(true);
  });

  it("filters edges by relation", () => {
    const edge: GraphEdge = {
      edge_id: "e1",
      source: "a",
      target: "b",
      relation: "SAME_SECTION",
      physical: true,
      logical: false,
      accepted: true,
      label: "SAME_SECTION",
      neo4j_type: "HAS_CHUNK",
    };
    const ids = new Set(["a", "b"]);
    expect(edgeVisible(edge, ids, filters)).toBe(true);
    expect(edgeVisible(edge, ids, { ...filters, relations: new Set(["ADJACENT"]) })).toBe(false);
  });
});

describe("node drawer and truncation", () => {
  it("shows every path and ignored reason", () => {
    const selected = node({
      node_id: "cand-1",
      path_ids: ["p1", "p2"],
      status: "accepted",
    });
    const trace = baseTrace({
      graph: {
        seeds: [],
        enabled_relations: [],
        disabled_relations: [],
        nodes: [selected],
        edges: [],
        paths: [
          {
            path_id: "p1",
            seed_chunk_id: "seed-1",
            relation: "SAME_SECTION",
            via: "Intro",
            candidate_chunk_id: "cand-1",
            accepted: true,
            reason_code: "SELECTED_GRAPH_CANDIDATE",
            reason_detail: "selected",
            logical: false,
            contribution: 1,
            seed_weight: 1,
          },
          {
            path_id: "p2",
            seed_chunk_id: "seed-1",
            relation: "ADJACENT",
            via: null,
            candidate_chunk_id: "cand-1",
            accepted: false,
            reason_code: "DUPLICATE_PATH",
            reason_detail: "already reached",
            logical: false,
            contribution: 0.5,
            seed_weight: 1,
          },
        ],
        truncated: false,
        truncation_caps: [],
        truncation_warning: "",
        selected_chunk_ids: ["cand-1"],
        counters: {},
      },
    });
    render(<NodeDrawer node={selected} trace={trace} onClose={() => undefined} />);
    expect(screen.getAllByTestId("graph-path")).toHaveLength(2);
    expect(screen.getByText(/DUPLICATE_PATH/)).toBeInTheDocument();
    expect(screen.getByText(/already reached/)).toBeInTheDocument();
  });

  it("shows a truncation warning", () => {
    const trace = baseTrace({
      graph: {
        seeds: [],
        enabled_relations: [],
        disabled_relations: [],
        nodes: [],
        edges: [],
        paths: [],
        truncated: true,
        truncation_caps: ["max_nodes"],
        truncation_warning: "Graph trace was truncated by a safety cap.",
        selected_chunk_ids: [],
        counters: {},
      },
    });
    render(<GraphPanel trace={trace} onSelectNode={() => undefined} selectedId={null} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/truncated/i);
  });
});

describe("fusion, reranker, evidence, generator", () => {
  it("renders fusion contributions", () => {
    const trace = baseTrace({
      fusion: {
        rrf_k: 60,
        weight_dense: 1,
        weight_fulltext: 1,
        weight_graph: 0.5,
        formula: "score(chunk) = Σ weight_c / (k + rank_c)",
        candidates: [
          {
            chunk_id: "c1",
            dense_rank: 1,
            dense_contribution: 0.016,
            lexical_rank: 2,
            lexical_contribution: 0.015,
            graph_rank: null,
            graph_contribution: 0,
            channels: ["dense", "fulltext"],
            rrf_score: 0.031,
            fused_rank: 1,
            text: "Weather can change from day to day",
            provenance: {},
          },
        ],
      },
    });
    render(<FusionPanel trace={trace} />);
    expect(screen.getByTestId("rrf-formula")).toHaveTextContent("k=60");
    expect(screen.getByTestId("fusion-row")).toHaveTextContent("0.016");
  });

  it("shows reranker rank movement", () => {
    const trace = baseTrace({
      reranker: {
        score_kind: "raw_relevance_logit",
        candidates: [
          {
            chunk_id: "c1",
            fused_rank: 3,
            reranked_rank: 1,
            rank_movement: 2,
            rerank_score: 1.4,
            text: "temperature, wind and clouds",
            provenance: {},
            survived_final_top_k: true,
          },
        ],
      },
    });
    render(<RerankerPanel trace={trace} />);
    expect(screen.getByTestId("rank-movement")).toHaveTextContent("+2");
    expect(screen.getByText(/raw BGE relevance logits/i)).toBeInTheDocument();
  });

  it("renders evidence pass/fail cards", () => {
    const trace = baseTrace({
      evidence: {
        sufficient: false,
        confidence: "insufficient",
        checks: [
          { name: "candidates_exist", passed: true, value: 2, threshold: 1, detail: "2 chunks" },
          { name: "query_term_overlap", passed: false, value: 0.05, threshold: 0.15, detail: "too low" },
        ],
        kept_chunk_ids: [],
        reasons: ["barely mentions"],
      },
    });
    render(<EvidencePanel trace={trace} />);
    expect(screen.getByTestId("evidence-verdict")).toHaveTextContent("INSUFFICIENT");
    expect(screen.getAllByTestId("evidence-check")).toHaveLength(2);
    expect(screen.getByText("FAIL")).toBeInTheDocument();
  });

  it("streams generator output and shows skipped generation", () => {
    const live = baseTrace({
      prompt: {
        tutor_state: "GIVE_HINT",
        system_prompt: "Be a tutor",
        user_prompt: "STUDENT QUESTION",
        evidence_blocks: [],
        generation_settings: { temperature: 0.7 },
        generation_skipped: false,
        skip_reason: "",
      },
      generation: { tokens: ["Hel", "lo"], response_text: "Hello", elapsed_ms: 10 },
    });
    const { rerender } = render(<LiveResponse trace={live} running />);
    expect(screen.getByTestId("generation-stream")).toHaveTextContent("Hello");
    const skipped = baseTrace({
      prompt: {
        tutor_state: "INSUFFICIENT_EVIDENCE",
        system_prompt: "sys",
        user_prompt: "usr",
        evidence_blocks: [],
        generation_settings: {},
        generation_skipped: true,
        skip_reason: "Retrieval-only mode: generation was not requested.",
      },
      stages: {
        ...emptyStages(),
        generator: { name: "generator", status: "skipped", elapsed_ms: 0, summary: "skipped", error: null },
      },
    });
    rerender(<LiveResponse trace={skipped} running={false} />);
    expect(screen.getByTestId("generation-skipped")).toHaveTextContent(/retrieval-only/i);
  });

  it("does not render textbook figures in the generator output", () => {
    render(
      <LiveResponse
        trace={baseTrace({
          attached_figures: [{ image_id: "p1:img01", page_number: 7 }],
          generation: { tokens: ["ok"], response_text: "ok", elapsed_ms: 1 },
        })}
        running={false}
      />,
    );
    expect(screen.queryByText(/figures in this answer/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
