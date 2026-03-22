import logging

import gurobipy as gp
import networkx as nx
from data_schema import Instance, Solution
from gurobipy import GRB



class MiningRoutingSolver:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.budget = instance.budget
        logging.info("Creating model ...")
        logging.info(
            "Instance has %d locations, %d mines, %d tunnels, and a budget of %.2f",
            len(instance.locations),
            len(instance.mines),
            len(instance.tunnels),
            instance.budget,
        )
        # TODO: Implement me!
        # create model
        self.model = gp.Model()

        # build directed graph 
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.instance.locations)
        for tunnel in self.instance.tunnels:
            self.graph.add_edge(tunnel.source, tunnel.target, 
                                capacity=tunnel.throughput_per_hour, 
                                cost=tunnel.reinforcement_costs)
            self.graph.add_edge(tunnel.target, tunnel.source, 
                                capacity=tunnel.throughput_per_hour, 
                                cost=tunnel.reinforcement_costs)
        # flow vars 
        self.flow = {(u,v): self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=self.graph[u][v]['capacity'], name=f"flow_{u}_{v}") for u,v in self.graph.edges}

        # reinforced vars 
        self.reinforced = {}
        for tunnel in instance.tunnels:
            u, v = tunnel.source, tunnel.target
            self.reinforced[(u, v)] = self.model.addVar(vtype=GRB.BINARY, name=f"reinforced_{u}_{v}")
            self.reinforced[(v, u)] = self.model.addVar(vtype=GRB.BINARY, name=f"reinforced_{v}_{u}")
        
        # budget constraint sum costs for all used directed edges
        budget = gp.quicksum(tunnel.reinforcement_costs * self.reinforced[(tunnel.source, tunnel.target)] 
                             + tunnel.reinforcement_costs * self.reinforced[(tunnel.target, tunnel.source)]
                             for tunnel in instance.tunnels)
        self.model.addConstr(budget <= self.budget)

        # flow only if edge is reinforced
        for tunnel in instance.tunnels:
            u, v = tunnel.source, tunnel.target
            self.model.addConstr(self.flow[u, v] <= tunnel.throughput_per_hour * self.reinforced[(u, v)])
            self.model.addConstr(self.flow[v, u] <= tunnel.throughput_per_hour * self.reinforced[(v, u)])
        
        # at most one direction per tunnel
        for tunnel in instance.tunnels:
            u, v = tunnel.source, tunnel.target
            self.model.addConstr(self.reinforced[(u, v)] + self.reinforced[(v, u)] <= 1)



        # flow conservation const 
        elevator = instance.elevator_location
        for node in self.graph.nodes:
            if node == elevator:
                for successor in self.graph.successors(node):
                    self.model.addConstr(self.flow[node, successor] == 0)
            else :
                in_flow = gp.quicksum(self.flow[pred, node] for pred in self.graph.predecessors(node))
                out_flow = gp.quicksum(self.flow[node, succ] for succ in self.graph.successors(node))
                production = instance.mines[node].ore_per_hour if node in instance.mines else 0
                self.model.addConstr(out_flow <= in_flow + production)

        # maximize flow into elevator
        obj = gp.quicksum(self.flow[pred, elevator] for pred in self.graph.predecessors(elevator))


        self.model.setObjective(obj, GRB.MAXIMIZE)


    


    def solve(self) -> Solution:
        """
        Calculate the optimal solution to the problem.
        Returns the "flow" as a list of tuples, each tuple with two entries:
            - The *directed* edge tuple. Both entries in the edge should be ints, representing the ids of locations.
            - The throughput/utilization of the edge, in goods per hour
        """
        # TODO: implement me!
        logging.info("Solving model...")
        self.model.optimize()
        if self.model.SolCount == 0:
            logging.info("sorrryyyy no solution found")
        else :
            logging.info("yayyyy solution found")
        flow_sol = [(( u, v), int(round(var.X))) for (u, v), var in self.flow.items() if var.X > 0.5]
        return Solution(flow=flow_sol)


