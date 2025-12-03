import math
from enum import Enum

import networkx as nx
from _timer import Timer
from solution_hamiltonian import HamiltonianCycleModel

import logging
logging.basicConfig(level=logging.INFO)
class SearchStrategy(Enum):
    """
    Different search strategies for the solver.
    """

    SEQUENTIAL_UP = 1  # Try smallest possible k first.
    SEQUENTIAL_DOWN = 2  # Try any improvement.
    BINARY_SEARCH = 3  # Try a binary search for the optimal k.

    def __str__(self):
        return self.name.title()

    @staticmethod
    def from_str(s: str):
        return SearchStrategy[s.upper()]


class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph) -> None:
        """
        Creates a solver for the Bottleneck Traveling Salesman Problem on the given networkx graph.
        You can assume that the input graph is complete, so all nodes are neighbors.
        The distance between two neighboring nodes is a numeric value (int / float), saved as
        an edge data parameter called "weight".
        There are multiple ways to access this data, and networkx also implements
        several algorithms that automatically make use of this value.
        Check the networkx documentation for more information!
        """
        self.graph = graph
        # TODO: Implement me!
        # exrcat all edges and sort them 
        weights = [self.graph.edges[e]["weight"] for e in self.graph.edges]
        self.edges_weights = sorted(set(weights))



    def lower_bound(self) -> float:
        # TODO: Implement me!
        # easy lazy soltuion, the bottelneck is the minimum edge weight
        return self.edges_weights[0] if self.edges_weights else 0.0


    def create_subgraph(self, threshold: float) -> nx.Graph:
        copy_graph = nx.Graph()
        copy_graph.add_nodes_from(self.graph.nodes)
        for u, v, weight in self.graph.edges(data="weight"):
            if weight <= threshold:
                copy_graph.add_edge(u, v)
        return copy_graph


    def optimize_bottleneck(
        self,
        time_limit: float = math.inf,
        search_strategy: SearchStrategy = SearchStrategy.BINARY_SEARCH,
    ) -> list[tuple[int, int]] | None:
        """
        Find the optimal bottleneck tsp tour.
        """

        self.timer = Timer(time_limit)
        # TODO: Implement me!

        
        # binary for the win 
        left, right = 0, len(self.edges_weights) - 1
        best_tour = None
        while left <= right:
            mid = (left + right) // 2
            threshold = self.edges_weights[mid] # edge weight to remove

            # create subgraph with edges <= threshold
            copy_graph = self.create_subgraph(threshold)
            if not nx.is_connected(copy_graph):
                left = mid + 1
                continue

            # solve it 
            hc_solver = HamiltonianCycleModel(copy_graph)
            hc_edges = hc_solver.solve()

            if hc_edges is not None:
                # we found a tour but can we go lower 
                best_tour = hc_edges
                right = mid - 1
            else:
                left = mid + 1
        return best_tour



