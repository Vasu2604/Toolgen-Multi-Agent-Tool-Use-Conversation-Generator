"""
graph.py — Tool Graph + Tool-Chain Sampler

Spec requires node types:
  Tool, Endpoint, Parameter, ResponseField (when available), Concept/Tag

Graph structure:
  Nodes:
    tool:{name}               — Tool node
    endpoint:{tool}::{ep}     — Endpoint node
    param:{tool}::{ep}::{p}   — Parameter node (required + optional)
    rf:{tool}::{ep}::{field}  — ResponseField node
    category:{cat}            — Category / Concept / Tag node

  Edges:
    Tool       → Endpoint      (has_endpoint)
    Endpoint   → Parameter     (has_param)
    Endpoint   → ResponseField (has_response_field)
    Tool       → Category      (belongs_to)
    Tool       ↔ Tool          (same_category)
    Endpoint   → Endpoint      (feeds_into — RF name matches param name)
"""

import json
import random
from pathlib import Path
from typing import Optional

import networkx as nx

from toolgen.registry import ToolRegistry, Tool, Endpoint


class ToolGraph:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.graph    = nx.DiGraph()
        self._build()

    # ── node-id helpers ───────────────────────────────────────
    @staticmethod
    def _tool_id(name: str)                       -> str: return f"tool:{name}"
    @staticmethod
    def _ep_id(tool: str, ep: str)                -> str: return f"endpoint:{tool}::{ep}"
    @staticmethod
    def _param_id(tool: str, ep: str, p: str)     -> str: return f"param:{tool}::{ep}::{p}"
    @staticmethod
    def _rf_id(tool: str, ep: str, field: str)    -> str: return f"rf:{tool}::{ep}::{field}"
    @staticmethod
    def _cat_id(cat: str)                         -> str: return f"category:{cat}"

    def _build(self):
        tools = self.registry.get_all_tools()

        # ── Tool nodes ─────────────────────────────────────────
        for tool in tools:
            self.graph.add_node(
                self._tool_id(tool.name),
                node_type="tool",
                name=tool.name,
                description=tool.description,
                category=tool.category,
            )

        # ── Category nodes ─────────────────────────────────────
        categories = {t.category for t in tools}
        for cat in categories:
            self.graph.add_node(
                self._cat_id(cat),
                node_type="category",
                name=cat,
            )

        for tool in tools:
            # Tool → Category
            self.graph.add_edge(
                self._tool_id(tool.name),
                self._cat_id(tool.category),
                relation="belongs_to",
            )

        # ── Same-category edges ────────────────────────────────
        for cat in categories:
            cat_tools = [t for t in tools if t.category == cat]
            for i, t1 in enumerate(cat_tools):
                for t2 in cat_tools[i + 1:]:
                    self.graph.add_edge(
                        self._tool_id(t1.name),
                        self._tool_id(t2.name),
                        relation="same_category",
                    )
                    self.graph.add_edge(
                        self._tool_id(t2.name),
                        self._tool_id(t1.name),
                        relation="same_category",
                    )

        # ── Endpoint + Parameter + ResponseField nodes ─────────
        for tool in tools:
            for ep in tool.endpoints:
                ep_node = self._ep_id(tool.name, ep.name)

                # Endpoint node
                self.graph.add_node(
                    ep_node,
                    node_type="endpoint",
                    tool_name=tool.name,
                    name=ep.name,
                    description=ep.description,
                    response_fields=ep.response_fields,
                )
                # Tool → Endpoint
                self.graph.add_edge(
                    self._tool_id(tool.name), ep_node, relation="has_endpoint"
                )

                # Parameter nodes (required + optional)
                for param in ep.required_params + ep.optional_params:
                    p_node = self._param_id(tool.name, ep.name, param.name)
                    self.graph.add_node(
                        p_node,
                        node_type="parameter",
                        name=param.name,
                        param_type=param.type,
                        required=param.required,
                        description=param.description,
                    )
                    # Endpoint → Parameter
                    self.graph.add_edge(ep_node, p_node, relation="has_param")

                # ResponseField nodes
                for field_name in ep.response_fields:
                    rf_node = self._rf_id(tool.name, ep.name, field_name)
                    self.graph.add_node(
                        rf_node,
                        node_type="response_field",
                        name=field_name,
                        endpoint_id=ep.endpoint_id,
                    )
                    # Endpoint → ResponseField
                    self.graph.add_edge(
                        ep_node, rf_node, relation="has_response_field"
                    )

        # ── feeds_into edges (RF name matches a param name) ────
        all_eps = self.registry.get_all_endpoints()
        for ep_a in all_eps:
            out_fields = set(ep_a.response_fields)
            ep_a_node  = self._ep_id(ep_a.tool_name, ep_a.name)
            for ep_b in all_eps:
                if ep_a.endpoint_id == ep_b.endpoint_id:
                    continue
                in_fields = {p.name for p in ep_b.required_params + ep_b.optional_params}
                overlap   = out_fields & in_fields
                if overlap:
                    ep_b_node = self._ep_id(ep_b.tool_name, ep_b.name)
                    self.graph.add_edge(
                        ep_a_node, ep_b_node,
                        relation="feeds_into",
                        shared_fields=list(overlap),
                    )

        n = self.graph.number_of_nodes()
        e = self.graph.number_of_edges()
        # Count by node type for summary
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("node_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(
            f"[Graph] Built graph: {n} nodes, {e} edges  "
            + "  ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
        )

    def save(self, path: str):
        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Graph] Saved to {path}")

    def get_chaining_endpoints(self, ep_node_id: str) -> list[str]:
        """Return endpoint node IDs that can follow ep_node_id (feeds_into)."""
        return [
            tgt for _, tgt, d in self.graph.out_edges(ep_node_id, data=True)
            if d.get("relation") == "feeds_into"
        ]

    def get_tool_neighbors(self, tool_name: str) -> list[str]:
        """Return tool names in same category."""
        tool_node = self._tool_id(tool_name)
        result = []
        for nbr in self.graph.neighbors(tool_node):
            d = self.graph.nodes.get(nbr, {})
            if d.get("node_type") == "tool":
                result.append(d["name"])
        return result


class ToolChainSampler:
    """
    Samples realistic tool chains from the Tool Graph.
    All sampling goes through the graph — no hardcoded lists.
    """

    def __init__(self, graph: ToolGraph, registry: ToolRegistry, seed: int = 42):
        self.graph    = graph
        self.registry = registry
        self.rng      = random.Random(seed)

    def _ep_node(self, ep_id: str) -> str:
        ep = self.registry.get_endpoint(ep_id)
        if not ep:
            return ""
        return ToolGraph._ep_id(ep.tool_name, ep.name)

    def sample_sequential_chain(self, min_tools: int = 3, max_tools: int = 5) -> list[str]:
        """
        Walk feeds_into / same_category edges to build a realistic chain.
        Returns list of endpoint_ids (registry IDs, not graph node IDs).
        """
        all_eps = self.registry.get_all_endpoints()
        if not all_eps:
            return []

        start      = self.rng.choice(all_eps)
        chain      = [start.endpoint_id]
        used_tools = {start.tool_name}
        target     = self.rng.randint(min_tools, max_tools)

        for _ in range(target - 1):
            last_ep_id   = chain[-1]
            last_ep_node = self._ep_node(last_ep_id)

            # Prefer feeds_into edges
            next_candidates = [
                ep_id for ep_id in [
                    # convert graph node IDs back to endpoint_ids
                    n.replace("endpoint:", "").replace("::", "::", 1)
                    for n in self.graph.get_chaining_endpoints(last_ep_node)
                ]
                if self.registry.get_endpoint(ep_id) and
                   self.registry.get_endpoint(ep_id).tool_name not in used_tools
            ]
            # Also accept same-category tools
            if not next_candidates:
                last_ep = self.registry.get_endpoint(last_ep_id)
                if last_ep:
                    same_cat = self.registry.get_tools_by_category(
                        self.registry.get_tool(last_ep.tool_name).category
                        if self.registry.get_tool(last_ep.tool_name) else "general"
                    )
                    next_candidates = [
                        e.endpoint_id
                        for t in same_cat if t.name not in used_tools
                        for e in t.endpoints
                    ]

            if not next_candidates:
                remaining = [
                    e.endpoint_id for e in all_eps
                    if e.tool_name not in used_tools
                ]
                if not remaining:
                    break
                next_candidates = remaining

            chosen = self.rng.choice(next_candidates)
            ep     = self.registry.get_endpoint(chosen)
            if ep:
                chain.append(chosen)
                used_tools.add(ep.tool_name)

        return chain

    def sample_parallel_chain(self, num_tools: int = 2) -> list[list[str]]:
        """Return a parallel batch (list of endpoint_ids called simultaneously)."""
        categories = list({t.category for t in self.registry.get_all_tools()})
        self.rng.shuffle(categories)
        for cat in categories:
            cat_tools = self.registry.get_tools_by_category(cat)
            if len(cat_tools) >= num_tools:
                chosen = self.rng.sample(cat_tools, num_tools)
                batch  = [self.rng.choice(t.endpoints).endpoint_id
                          for t in chosen if t.endpoints]
                if batch:
                    return [batch]
        all_eps = self.registry.get_all_endpoints()
        chosen  = self.rng.sample(all_eps, min(num_tools, len(all_eps)))
        return [[ep.endpoint_id for ep in chosen]]

    def sample_chain(self, pattern: str = "sequential") -> list:
        """
        Main entry. All patterns guarantee ≥ 3 endpoint_ids total.
        pattern: "sequential" | "parallel" | "mixed"
        """
        if pattern == "parallel":
            par      = self.sample_parallel_chain(num_tools=2)
            flat_par = par[0] if par else []
            par_tools = {
                self.registry.get_endpoint(e).tool_name
                for e in flat_par if self.registry.get_endpoint(e)
            }
            seq = [
                e for e in self.sample_sequential_chain(min_tools=1, max_tools=2)
                if self.registry.get_endpoint(e) and
                   self.registry.get_endpoint(e).tool_name not in par_tools
            ]
            return flat_par + seq[:1]

        elif pattern == "mixed":
            seq = self.sample_sequential_chain(min_tools=2, max_tools=3)
            par = self.sample_parallel_chain(num_tools=2)
            return seq + (par[0] if par else [])

        else:
            return self.sample_sequential_chain()

    def get_chain_tools(self, chain: list) -> list[str]:
        seen, tools = set(), []
        for item in chain:
            items = item if isinstance(item, list) else [item]
            for ep_id in items:
                ep = self.registry.get_endpoint(ep_id)
                if ep and ep.tool_name not in seen:
                    seen.add(ep.tool_name)
                    tools.append(ep.tool_name)
        return tools
