import { useEffect, useMemo, useRef, useState } from "react";
import cytoscape from "cytoscape";
import dagre from "cytoscape-dagre";
import type { GraphNode, GraphTrace, RunTrace } from "../types";
import {
  ROLE_COLORS,
  cytoscapeStyle,
  edgeVisible,
  nodeVisible,
  visualRoleForNode,
  type GraphFilterState,
} from "../graphStyles";

cytoscape.use(dagre);

interface Props {
  trace: RunTrace | null;
  onSelectNode: (node: GraphNode | null) => void;
  selectedId: string | null;
}

function topologyKey(graph: GraphTrace | null, filters: GraphFilterState): string {
  if (!graph) return "";
  return [
    graph.nodes.map((node) => node.node_id).join(","),
    graph.edges.map((edge) => edge.edge_id).join(","),
    String(filters.showContext),
    String(filters.showAccepted),
    String(filters.showIgnored),
    [...filters.relations].sort().join(","),
    filters.search,
  ].join("|");
}

function graphElements(graph: GraphTrace, filters: GraphFilterState): cytoscape.ElementDefinition[] {
  const visibleNodes = graph.nodes.filter((node) => nodeVisible(node, filters));
  const visibleIds = new Set(visibleNodes.map((node) => node.node_id));
  const elements: cytoscape.ElementDefinition[] = visibleNodes.map((node) => {
    const role = visualRoleForNode(node);
    const colors = ROLE_COLORS[role];
    return {
      group: "nodes",
      data: {
        id: node.node_id,
        label: node.display_label,
        kind: node.node_kind,
        role,
        background: colors.background,
        border: colors.border,
        textColor: colors.text,
      },
    };
  });
  for (const edge of graph.edges) {
    if (!edgeVisible(edge, visibleIds, filters)) continue;
    elements.push({
      group: "edges",
      data: {
        id: edge.edge_id,
        source: edge.source,
        target: edge.target,
        label: edge.logical ? `${edge.relation} (logical)` : edge.relation,
        accepted: edge.accepted ? "yes" : "no",
        logical: edge.logical ? "yes" : "no",
      },
    });
  }
  return elements;
}

const DAGRE: cytoscape.LayoutOptions = {
  name: "dagre",
  rankDir: "LR",
  nodeSep: 24,
  rankSep: 48,
  fit: false,
  animate: false,
} as cytoscape.LayoutOptions;

export function GraphPanel({ trace, onSelectNode, selectedId }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const graphRef = useRef<GraphTrace | null>(null);
  const onSelectRef = useRef(onSelectNode);
  const graph = trace?.graph ?? null;
  graphRef.current = graph;
  onSelectRef.current = onSelectNode;

  const relations = useMemo(() => {
    const names = new Set<string>();
    for (const edge of graph?.edges ?? []) names.add(edge.relation);
    return [...names].sort();
  }, [graph]);

  const [filters, setFilters] = useState<GraphFilterState>({
    showContext: true,
    showAccepted: true,
    showIgnored: true,
    relations: new Set(),
    search: "",
  });

  const layoutKey = topologyKey(graph, filters);
  const colorKey = (graph?.nodes ?? [])
    .map((node) => `${node.node_id}:${visualRoleForNode(node)}`)
    .join("|");

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      style: cytoscapeStyle() as cytoscape.Stylesheet[],
      layout: { name: "preset" },
      minZoom: 0.15,
      maxZoom: 8,
      wheelSensitivity: 3.5,
      userZoomingEnabled: true,
      userPanningEnabled: true,
    });
    cy.on("tap", "node", (event) => {
      const id = event.target.id();
      const node = (graphRef.current?.nodes ?? []).find((item) => item.node_id === id) ?? null;
      onSelectRef.current(node);
    });
    cy.on("tap", (event) => {
      if (event.target === cy) onSelectRef.current(null);
    });
    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  const filtersRef = useRef(filters);
  filtersRef.current = filters;

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const currentGraph = graphRef.current;
    if (!currentGraph) {
      cy.elements().remove();
      return;
    }
    cy.elements().remove();
    cy.add(graphElements(currentGraph, filtersRef.current));
    cy.layout(DAGRE).run();
    cy.fit(undefined, 24);
  }, [layoutKey]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !graph) return;
    for (const node of graph.nodes) {
      const ele = cy.getElementById(node.node_id);
      if (!ele.length) continue;
      const role = visualRoleForNode(node);
      const colors = ROLE_COLORS[role];
      ele.data({
        label: node.display_label,
        kind: node.node_kind,
        role,
        background: colors.background,
        border: colors.border,
        textColor: colors.text,
      });
    }
  }, [graph, colorKey]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().unselect();
    if (selectedId && cy.getElementById(selectedId).length) {
      cy.getElementById(selectedId).select();
    }
  }, [selectedId]);

  const counters = graph?.counters ?? {};

  return (
    <section className="graph-panel">
      <header className="panel-header">
        <h2>Graph expansion</h2>
        {graph?.truncated ? (
          <p className="truncation-warning" role="alert">
            {graph.truncation_warning || "Graph trace was truncated by a safety cap."}
          </p>
        ) : null}
      </header>
      <div className="graph-toolbar">
        <input
          placeholder="Search node ID or label"
          value={filters.search}
          onChange={(e) => setFilters({ ...filters, search: e.target.value })}
        />
        <label>
          <input
            type="checkbox"
            checked={filters.showContext}
            onChange={(e) => setFilters({ ...filters, showContext: e.target.checked })}
          />
          Context
        </label>
        <label>
          <input
            type="checkbox"
            checked={filters.showAccepted}
            onChange={(e) => setFilters({ ...filters, showAccepted: e.target.checked })}
          />
          Accepted
        </label>
        <label>
          <input
            type="checkbox"
            checked={filters.showIgnored}
            onChange={(e) => setFilters({ ...filters, showIgnored: e.target.checked })}
          />
          Ignored
        </label>
        {relations.map((relation) => (
          <label key={relation}>
            <input
              type="checkbox"
              checked={filters.relations.size === 0 || filters.relations.has(relation)}
              onChange={(e) => {
                const next = new Set(filters.relations);
                if (e.target.checked) next.add(relation);
                else {
                  if (next.size === 0) relations.forEach((r) => next.add(r));
                  next.delete(relation);
                }
                setFilters({ ...filters, relations: next.size === relations.length ? new Set() : next });
              }}
            />
            {relation}
          </label>
        ))}
        <button type="button" onClick={() => cyRef.current?.fit()}>
          Fit
        </button>
        <button
          type="button"
          onClick={() => {
            const cy = cyRef.current;
            if (!cy) return;
            cy.layout(DAGRE).run();
            cy.fit(undefined, 24);
          }}
        >
          Reset layout
        </button>
      </div>
      <div className="graph-canvas" ref={containerRef} data-testid="graph-canvas" />
      <ul className="legend">
        <li><span className="swatch seed" /> Seed</li>
        <li><span className="swatch accepted" /> Accepted graph</li>
        <li><span className="swatch evidence" /> Final evidence</li>
        <li><span className="swatch mismatch" /> Metadata mismatch</li>
        <li><span className="swatch ignored" /> Ignored</li>
        <li><span className="swatch hub" /> Hub concept</li>
        <li><span className="swatch context" /> Context</li>
        <li>Dashed = ignored path · Dotted cyan = logical</li>
      </ul>
      <dl className="counters">
        <div><dt>Seeds</dt><dd>{counters.seeds ?? graph?.seeds.length ?? 0}</dd></div>
        <div><dt>Context</dt><dd>{counters.context_nodes ?? 0}</dd></div>
        <div><dt>Examined chunks</dt><dd>{counters.candidate_chunks_examined ?? 0}</dd></div>
        <div><dt>Accepted</dt><dd>{counters.accepted_graph_candidates ?? 0}</dd></div>
        <div><dt>Ignored</dt><dd>{counters.ignored_candidates ?? 0}</dd></div>
        <div><dt>Rejected paths</dt><dd>{counters.rejected_paths ?? 0}</dd></div>
        <div><dt>Truncated</dt><dd>{graph?.truncated ? "yes" : "no"}</dd></div>
      </dl>
      <DisabledRelations graph={graph} />
    </section>
  );
}

function DisabledRelations({ graph }: { graph: GraphTrace | null }) {
  const disabled = graph?.disabled_relations ?? [];
  if (!disabled.length) return null;
  return (
    <p className="disabled-relations">
      Disabled by depth:{" "}
      {disabled.map((item) => item.relation).join(", ")}. Neighbouring nodes were not examined.
    </p>
  );
}
