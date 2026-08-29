from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphNode:
    name: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


@dataclass
class RailGraph:
    """
    Model-level topology graph: nodes are compiled tensors + ops.
    """

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def add_node(self, node: GraphNode):
        self.nodes[node.name] = node

    def add_edge(self, src: str, dst: str):
        self.edges.append((src, dst))

    def to_dict(self) -> dict:
        return {
            "nodes": {
                k: {"op": v.op, "inputs": v.inputs, "attrs": v.attrs} for k, v in self.nodes.items()
            },
            "edges": self.edges,
        }
