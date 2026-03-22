import networkx as nx
import random
from .greedy import greedy_coloring, num_colors


def multistart_greedy(graph: nx.Graph, iterations: int = 50, seed: int = None) -> dict:
    if seed is not None:
        random.seed(seed)
    
    nodes = list(graph.nodes())
    best_coloring = None
    best_num_colors = float('inf')
    
    for _ in range(iterations):
        random.shuffle(nodes)
        coloring = greedy_coloring(graph, order=nodes)
        colors_used = num_colors(coloring)
        
        if colors_used < best_num_colors:
            best_coloring = coloring.copy()
            best_num_colors = colors_used

    return best_coloring

