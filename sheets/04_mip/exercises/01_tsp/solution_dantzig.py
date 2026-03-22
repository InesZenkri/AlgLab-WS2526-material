"""
Implement the Dantzig-Fulkerson-Johnson formulation for the TSP.
"""

import logging
import typing

import gurobipy as gp
import networkx as nx


class GurobiTspSolver:
    """
    IMPLEMENT ME!
    """

    def __init__(self, G: nx.Graph, k: int = 2):
        """
        G is a weighted networkx graph, where the weight of an edge is stored in the
        "weight" attribute. It is strictly positive.
        """
        self.graph = G
        assert (
            G.number_of_edges() == G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        ), "Invalid graph"
        assert all(
            weight > 0
            for _, _, weight in G.edges.data("weight", default=None)  # type: ignore[attr-defined]
        ), "Invalid graph"
        assert k in {1, 2}, "Invalid k"
        self.k = k
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
            self.vars[key] = self._model.addVar(vtype=gp.GRB.BINARY, name=f"x_{u}_{v}")
        
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
        """
        # TODO: Implement me!
        if self._model.SolCount == 0:
            return None
        solution_edges = []
        for (u,v), val in self.vars.items():
            if val.X > 0.5:
                solution_edges.append((u, v))
        
        return nx.Graph(solution_edges)

    def get_objective(self) -> typing.Optional[float]:
        """
        Return the objective value of the last solution.
        """
        # TODO: Implement me!
        if self._model.SolCount == 0:
            return None
        return round(self._model.ObjVal)

    def solve(self, time_limit: float, opt_tol: float = 0.001) -> None:
        """
        Solve the model. After solving the model, the solution, its objective value,
        and the lower bounds should be available via the corresponding methods.
        """
        logging.info("Solving model ...")
        # Set parameters for the solver.
        self._model.Params.LogToConsole = 1
        self._model.Params.TimeLimit = time_limit
        self._model.Params.LazyConstraints = 1
        self._model.Params.MIPGap = (
            opt_tol  # https://www.gurobi.com/documentation/11.0/refman/mipgap.html
        )

        # ...
        # TODO: Implement me!
        def subtour_elimination(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(self.vars)
                edges = [(u, v) for (u, v), val in vals.items() if val > 0.5]
                solution_graph = nx.Graph(edges)
                components = list(nx.connected_components(solution_graph))
                if len(components) > 1:
                    for comp in components:
                        outgoing_edges = [self.vars[tuple(sorted((u,v)))] for u in comp for v in self.graph.neighbors(u) if v not in comp]
                        model.cbLazy(gp.quicksum(outgoing_edges) >= self.k)
        self._model.optimize(subtour_elimination)


