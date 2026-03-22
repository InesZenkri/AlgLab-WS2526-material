import networkx as nx
from ortools.sat.python import cp_model


def solve_cp_alldiff(graph: nx.Graph, time_limit: float = 60.0, 
                     upper_bound: int = None, max_cliques: int = 100) -> dict:
    if len(graph) == 0:
        return {'coloring': {}, 'num_colors': 0, 'lower_bound': 0, 'status': 'optimal'}
    
    if upper_bound is None:
        upper_bound = max(dict(graph.degree()).values()) + 1
    
    model = cp_model.CpModel()
    
    z = {v: model.NewIntVar(0, upper_bound - 1, f'z_{v}') for v in graph.nodes()}
    z_max = model.NewIntVar(0, upper_bound - 1, 'z_max')
    
    cliques = sorted(nx.find_cliques(graph), key=len, reverse=True)[:max_cliques]
    for clique in cliques:
        if len(clique) >= 2:
            model.AddAllDifferent([z[v] for v in clique])
    
    for u, v in graph.edges():
        model.Add(z[u] != z[v])
    
    for v in graph.nodes():
        model.Add(z[v] <= z_max)
    
    model.Minimize(z_max)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    
    result = {'coloring': None, 'num_colors': None, 'lower_bound': None, 'status': None}
    
    if status == cp_model.OPTIMAL:
        result['status'] = 'optimal'
        result['coloring'] = {v: solver.Value(z[v]) for v in graph.nodes()}
        result['num_colors'] = solver.Value(z_max) + 1
        result['lower_bound'] = solver.Value(z_max) + 1
    elif status == cp_model.FEASIBLE:
        result['status'] = 'feasible'
        result['coloring'] = {v: solver.Value(z[v]) for v in graph.nodes()}
        result['num_colors'] = solver.Value(z_max) + 1
        result['lower_bound'] = int(solver.BestObjectiveBound()) + 1
    elif status == cp_model.INFEASIBLE:
        result['status'] = 'infeasible'
    else:
        result['status'] = 'timeout'
        if solver.BestObjectiveBound() is not None:
            result['lower_bound'] = int(solver.BestObjectiveBound()) + 1
    
    return result
