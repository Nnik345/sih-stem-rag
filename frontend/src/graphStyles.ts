import type { GraphEdge, GraphNode } from "./types";

export type VisualRole =
  | "seed"
  | "accepted"
  | "evidence"
  | "metadata_mismatch"
  | "ignored"
  | "hub"
  | "context";

export function visualRoleForNode(node: GraphNode): VisualRole {
  if (node.node_kind === "context" || node.status === "context") return "context";
  if (node.node_kind === "hub" || node.reason_codes.includes("HUB_CONCEPT_EXCLUDED")) return "hub";
  if (node.final_evidence || node.status === "evidence" || node.metadata.visual === "evidence") {
    return "evidence";
  }
  if (node.node_kind === "seed" || node.status === "seed") return "seed";
  if (node.metadata.visual === "metadata_mismatch" || node.reason_codes.includes("METADATA_FILTER_MISMATCH")) {
    return "metadata_mismatch";
  }
  if (node.status === "accepted") return "accepted";
  return "ignored";
}

export const ROLE_COLORS: Record<VisualRole, { background: string; border: string; text: string }> = {
  seed: { background: "#2563eb", border: "#1d4ed8", text: "#ffffff" },
  accepted: { background: "#7c3aed", border: "#6d28d9", text: "#ffffff" },
  evidence: { background: "#16a34a", border: "#15803d", text: "#ffffff" },
  metadata_mismatch: { background: "#f59e0b", border: "#d97706", text: "#1c1917" },
  ignored: { background: "#6b7280", border: "#4b5563", text: "#f9fafb" },
  hub: { background: "#1f2937", border: "#ef4444", text: "#fecaca" },
  context: { background: "#f8fafc", border: "#94a3b8", text: "#0f172a" },
};

export interface GraphFilterState {
  showContext: boolean;
  showAccepted: boolean;
  showIgnored: boolean;
  relations: Set<string>;
  search: string;
}

export function nodeVisible(node: GraphNode, filters: GraphFilterState): boolean {
  const role = visualRoleForNode(node);
  if (role === "context" && !filters.showContext) return false;
  if (role === "seed") {
    // Seeds stay visible so expansion origin is never lost.
  } else if ((role === "accepted" || role === "evidence") && !filters.showAccepted) {
    return false;
  }
  if ((role === "ignored" || role === "metadata_mismatch" || role === "hub") && !filters.showIgnored) {
    return false;
  }
  if (filters.search.trim()) {
    const q = filters.search.trim().toLowerCase();
    const hay = `${node.node_id} ${node.display_label} ${node.text}`.toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}

export function edgeVisible(
  edge: GraphEdge,
  visibleIds: Set<string>,
  filters: GraphFilterState,
): boolean {
  if (!visibleIds.has(edge.source) || !visibleIds.has(edge.target)) return false;
  if (filters.relations.size > 0 && !filters.relations.has(edge.relation)) return false;
  return true;
}

export function cytoscapeStyle(): object[] {
  return [
    {
      selector: "node",
      style: {
        label: "data(label)",
        "font-size": 10,
        "text-wrap": "ellipsis",
        "text-max-width": 90,
        "text-valign": "center",
        color: "data(textColor)",
        "background-color": "data(background)",
        "border-color": "data(border)",
        "border-width": 2,
        width: 28,
        height: 28,
      },
    },
    {
      selector: "node[kind = 'context']",
      style: { shape: "round-rectangle", width: 36, height: 22 },
    },
    {
      selector: "node[role = 'hub']",
      style: { "border-width": 4, "border-style": "solid" },
    },
    {
      selector: "edge",
      style: {
        width: 1.5,
        "line-color": "#64748b",
        "target-arrow-color": "#64748b",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        label: "data(label)",
        "font-size": 8,
        color: "#94a3b8",
        "text-rotation": "autorotate",
      },
    },
    {
      selector: "edge[accepted = 'yes']",
      style: { "line-style": "solid", opacity: 1 },
    },
    {
      selector: "edge[accepted = 'no']",
      style: { "line-style": "dashed", opacity: 0.45 },
    },
    {
      selector: "edge[logical = 'yes']",
      style: {
        "line-style": "dotted",
        "line-color": "#22d3ee",
        "target-arrow-color": "#22d3ee",
        label: "data(label)",
      },
    },
  ];
}
