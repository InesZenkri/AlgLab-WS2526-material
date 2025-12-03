import itertools

import networkx as nx
from pysat.solvers import Solver as SATSolver


class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.solver = SATSolver("Minicard")
        self.assumptions = []
        # TODO: Implement me!
        # edge (u,v) is the same as (v,u) so sort the endge 

        self.vars = {}
        self.edges = {}
        for i, (u, v) in enumerate(self.graph.edges, start=1):
            key = tuple(sorted((u, v)))  
            self.vars[key] = i # sat variable to edge
            self.edges[i] = key # edge to sat variable

    def degree_exact_two_constraint(self) -> None:
        for n in self.graph.nodes:
            incident_edges = [self.vars[tuple(sorted((n, u)))] for u in self.graph.neighbors(n)]
            negated_edges = [-x for x in incident_edges]

            # exact two edges incident to each node
            # at most two edges incident to each node
            self.solver.add_atmost(incident_edges, 2)
            # at least two edges incident to each node 
            # equivalent to at most len(incident_edges) - 2 edges incident to each node
            self.solver.add_atmost(negated_edges, len(incident_edges) - 2)


    def no_subtours_constraint(self, sets) -> None:
        """
        S = set(sets)
        existing_edges = []
        for u in S:
            for v in self.graph.neighbors(u):
                if v not in S:
                    key = tuple(sorted((u, v)))
                    edge_var = self.vars[key]
                    if edge_var not in existing_edges:
                        existing_edges.append(edge_var)

        # at least one edge must exist exiting the subset
        # equivalent to a clause x1 + x2 + ... + xn >= 1
        if existing_edges:
            self.solver.add_clause(existing_edges)
        """
        for S in sets:
            if len(S) == len(self.graph):
                continue
            
            edges = []
            for v in S:
                for u in self.graph.nodes:
                    if u not in S:
                        key = tuple(sorted((v, u)))
                        if key in self.vars:
                            edges.append(self.vars[key])
            
            neg_edges = [-x for x in edges]
            # at least one edge must exist exiting the subset
            # equivalnet to at most len(edges) - 1 edges exiting the subset
            self.solver.add_atmost(neg_edges, len(edges) - 1)

    
    def solve(self) -> list[tuple[int, int]] | None:
        """
        Solves the Hamiltonian Cycle Problem. If a HC is found,
        its edges are returned as a list.
        If the graph has no HC, 'None' is returned.
        """
        # TODO: Implement me!
        self.degree_exact_two_constraint()

        while True:
            # try solve with current constraints
            if not self.solver.solve():
                return None
            model = {i for i in self.solver.get_model() if i > 0} # only positive literals

            # extract edges that are part of the solution
            selected_edges = [self.edges[i] for i in model]
            tour_graph = nx.Graph()
            tour_graph.add_edges_from(selected_edges)
            tour_graph.add_nodes_from(self.graph.nodes)

            # find connected subsets in the tour graph
            sets = list(nx.connected_components(tour_graph))
            # if there is only one subset, we have a Hamiltonian cycle ezzz
            if len(sets) == 1:
                return selected_edges

            # otherwise, add subtour elimination constraints 
            self.no_subtours_constraint(sets)
           
    

