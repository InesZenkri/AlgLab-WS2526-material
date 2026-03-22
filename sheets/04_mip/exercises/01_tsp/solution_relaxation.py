"""
Implement the Dantzig-Fulkerson-Johnson formulation for the TSP.
"""

import logging
import typing

import gurobipy as gp
import networkx as nx

class GurobiTspRelaxationSolver:
    """
    IMPLEMENT ME!
    """

    def __init__(self, G: nx.Graph, k: int = 2):
        """
        G is a weighted networkx graph, where the weight of an edge is stored in the
        "weight" attribute. It is strictly positive.
        """
        self.graph = G
        self.k = k
        assert (
            G.number_of_edges() == G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        ), "Invalid graph"
        assert all(
            weight > 0
            for _, _, weight in G.edges.data("weight", default=None)  # type: ignore[attr-defined]
        ), "Invalid graph"
        assert k in {1, 2}, "Invalid k"
        logging.info("Creating model ...")
        logging.info(
            "Graph has %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        logging.info("Implementing subtour elimination with >= %d", k)
        self._model = gp.Model()
        # TODO: Implement me!
        # create variables for each edge
        self.vars = {}
        for u, v in self.graph.edges:
            key = tuple(sorted((u, v)))
            self.vars[key] = self._model.addVar(vtype=gp.GRB.CONTINUOUS,lb = 0.0, ub = 1.0, name=f"x_{u}_{v}")
        # objective is to minimize the total weight of the tour
        self._model.setObjective(
                gp.quicksum(
                    self.vars[tuple(sorted((u, v)))] * self.graph[u][v]["weight"] 
                    for u, v in self.graph.edges
                ), 
                gp.GRB.MINIMIZE
            )
        # add degree constraints
        for node in self.graph.nodes:
            incident_edges = [self.vars[tuple(sorted((node, neighbor)))] for neighbor in self.graph.neighbors(node)]
            self._model.addConstr(gp.quicksum(incident_edges) == 2)
    def get_lower_bound(self) -> float:
        """
        Return the current lower bound.
        """
        # TODO: Implement me!
        return self._model.ObjBound

    def get_solution(self) -> typing.Optional[nx.Graph]:
        """
        Return the current solution as a graph.

        The solution should be a networkx Graph were the
        fractional value of the edge is stored in the "x" attribute.
        You do not have to add edges with x=0.

        ```python
        graph = nx.Graph()
        graph.add_edge(0, 1, x=0.5)
        graph.add_edge(1, 2, x=1.0)
        ```
        """
        # TODO: Implement me!
        if self._model.SolCount == 0:
            return None
        solution = nx.Graph()
        for (u,v), val in self.vars.items():
            if val.X > 0.01:
                solution.add_edge(u, v, x=val.X)
        return solution

    def get_objective(self) -> typing.Optional[float]:
        """
        Return the objective value of the last solution.
        """
        # TODO: Implement me!
        if self._model.SolCount == 0:
            return None
        return self._model.ObjVal

    def solve(self) -> None:
        """
        Solve the model. After solving the model, the solution, its objective value,
        and the lower bounds should be available via the corresponding methods.
        """
        logging.info("Solving model ...")
        # Set parameters for the solver.
        self._model.Params.LogToConsole = 1

        # TODO: Implement me!
        max_iterations = 1000
        for iteration in range(max_iterations):
            self._model.optimize()
            if self._model.Status != gp.GRB.OPTIMAL:
                logging.warning("No optimal solution found after %d iterations", iteration)
                break
            solution_edges = [(u, v) for (u, v), val in self.vars.items() if val.X > 0.01]
            solution_graph = nx.Graph(solution_edges)
            components = list(nx.connected_components(solution_graph))
            if len(components) == 1:
                logging.info("Optimal solution found after %d iterations", iteration)
                break
            constraints_added = 0
            for comp in components:
                outgoing_edges = [self.vars[tuple(sorted((u,v)))] for u in comp for v in self.graph.neighbors(u) if v not in comp]
                current_sum = sum(var.X for var in outgoing_edges)
                if current_sum < self.k - 0.001:
                    self._model.addConstr(gp.quicksum(outgoing_edges) >= self.k)
                    constraints_added += 1
            if constraints_added == 0:
                logging.info("No constraints added after %d iterations", iteration)
                break
            else:
                logging.info("Optimal solution found after %d iterations", iteration)
