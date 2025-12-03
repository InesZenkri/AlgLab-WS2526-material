import bisect
import logging
import math
from typing import Iterable

import networkx as nx
from pysat.solvers import Solver as SATSolver

logging.basicConfig(level=logging.INFO)

# Define the node ID type. It is an integer but this helps to make the code more readable.
NodeId = int


class Distances:
    """
    This class provides a convenient interface to query distances between nodes in a graph.
    All distances are precomputed and stored in a dictionary, making lookups efficient.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self._distances = dict(nx.all_pairs_dijkstra_path_length(self.graph))

    def all_vertices(self) -> Iterable[NodeId]:
        """Returns an iterable of all node IDs in the graph."""
        return self._distances.keys()

    def dist(self, u: NodeId, v: NodeId) -> float:
        """Returns the distance between nodes `u` and `v`."""
        return self._distances[u].get(v, math.inf)

    def max_dist(self, centers: Iterable[NodeId]) -> float:
        """Returns the maximum distance from any node to the closest center."""
        return max(min(self.dist(c, u) for c in centers) for u in self.all_vertices())

    def vertices_in_range(self, u: NodeId, limit: float) -> Iterable[NodeId]:
        """Returns an iterable of nodes within `limit` distance from node `u`."""
        return (v for v, d in self._distances[u].items() if d <= limit)

    def sorted_distances(self) -> list[float]:
        """Returns a sorted list of all pairwise distances in the graph."""
        return sorted(
            dist
            for dist_dict in self._distances.values()
            for dist in dist_dict.values()
        )


class KCenterDecisionVariant:
    def __init__(self, distances: Distances, k: int) -> None:
        self.distances = distances
        # TODO: Implement me! 
        self.k = k
        self.solver = SATSolver("Gluecard4")
        self.all_vertices = list(self.distances.all_vertices())
        self.vars = {}    # node to sat variable
        self.nodes = {}  # sat var to node
        
        for i, node in enumerate(self.all_vertices, start=1):
            self.vars[node] = i
            self.nodes[i] = node
        
        #  at most k centers
        self.solver.add_atmost(lits=list(self.vars.values()), k=k)
        
        # Solution model
        self._solution: list[NodeId] | None = None

    def limit_distance(self, limit: float) -> None:
        """Adds constraints to the SAT solver to ensure coverage within the given distance."""
        logging.info("Limiting to distance: %f", limit)
        # TODO: Implement me!
        # each node must have one center in its range
        for node in self.distances.all_vertices():
            nearby_nodes = list(self.distances.vertices_in_range(node, limit))
            if len(nearby_nodes) == 0:
                self.solver.add_clause([])
                continue
            nearby_vars = [self.vars[n] for n in nearby_nodes]
            self.solver.add_clause(nearby_vars)

    def solve(self) -> list[NodeId] | None:
        """Solves the SAT problem and returns the list of selected nodes, if feasible."""
        # TODO: Implement me!
        if self.solver.solve():
            model = self.solver.get_model()
            self._solution = [node for node, var in self.vars.items() if model[var-1] > 0]
        else:
            self._solution = None
        return self._solution

    def get_solution(self) -> list[NodeId]:
        """Returns the solution if available; raises an error otherwise."""
        if self._solution is None:
            msg = "No solution available. Ensure `solve` is called first."
            raise ValueError(msg)
        return self._solution




class KCentersSolver:
    def __init__(self, graph: nx.Graph) -> None:
        """
        Creates a solver for the k-centers problem on the given networkx graph.
        The graph may not be complete, and edge weights are used to represent distances.
        """
        self.graph = graph
        # TODO: Implement me!

        self.distances = Distances(graph)


    def solve_heur(self, k: int) -> list[NodeId]:
        """
        Calculate a heuristic solution to the k-centers problem.
        Returns the k selected centers as a list of node IDs.
        """
        # TODO: Implement me!
        centers = []
        nodes_list = list(self.distances.all_vertices())
        
        for i in range(k):
            if i == 0:
                centers.append(nodes_list[0])  #just choose the first node, could be using random also 
            else:
                best_node = None #farthest node from the centers
                best_dist = -1 #distance to the farthest node from the centers(aka max distance)
                
                for node in nodes_list:
                    if node not in centers:
                        # distance to nearest existing center
                        dist_to_nearest = min(self.distances.dist(node, c) for c in centers)
                        if dist_to_nearest > best_dist:
                            best_dist = dist_to_nearest
                            best_node = node
                
                centers.append(best_node)
        return centers


    def solve(self, k: int) -> list[NodeId]:
        """
        Calculate the optimal solution to the k-centers problem for the given k.
        Returns the selected centers as a list of node IDs.
        """
        # Start with a heuristic solution
        centers = self.solve_heur(k)
        obj = self.distances.max_dist(centers)

        # TODO: Implement me!

        sorted_dists = self.distances.sorted_distances()
        possible_distances = [d for d in sorted_dists if d < obj]
        if not possible_distances:
            logging.info("welll Heuristic solution is optimal")
            return centers
        low = 0
        high = len(possible_distances) - 1
        best_centers = centers
        best_obj = obj
        while low <= high:
            mid = (low + high) // 2
            distance = possible_distances[mid]
            decision_solver = KCenterDecisionVariant(self.distances, k)
            decision_solver.limit_distance(distance)
            res = decision_solver.solve()
            if res is not None:
                logging.info("welllll new solution found, we can go lower")
                best_centers = res
                high = mid - 1
            else:
                logging.info("oppps too tight, we should go higher")
                low = mid + 1
        return best_centers
        

