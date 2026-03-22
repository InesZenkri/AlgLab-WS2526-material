
import networkx as nx


def dsatur(graph: nx.Graph) -> dict:
    coloring = {}
    saturation = {v: set() for v in graph.nodes()}
    n = len(graph)

    while len(coloring) < n:
        uncolored = [v for v in graph.nodes() if v not in coloring]
        best_vertex = max(
            uncolored,
            key=lambda v: (len(saturation[v]), graph.degree(v))
        )

        forbidden_colors = saturation[best_vertex]
        color = 0
        while color in forbidden_colors:
            color += 1

        coloring[best_vertex] = color

        for neighbor in graph.neighbors(best_vertex):
            saturation[neighbor].add(color)

    return coloring


def num_colors(coloring: dict) -> int:
    if not coloring:
        return 0
    return max(coloring.values()) + 1
