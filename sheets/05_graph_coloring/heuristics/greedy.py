import networkx as nx


def greedy_coloring(graph: nx.Graph, order: list = None) -> dict:
    if order is None:
        order = list(graph.nodes())
    
    coloring = {}
    
    for vertex in order:
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])
        
        color = 0
        while color in neighbor_colors:
            color += 1

        coloring[vertex] = color
    
    return coloring


def num_colors(coloring: dict) -> int:
    if not coloring:
        return 0
    return max(coloring.values()) + 1

