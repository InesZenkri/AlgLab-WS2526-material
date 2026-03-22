import networkx as nx

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from ortools.sat.python import cp_model


def solve_ass_cpsat(graph: nx.Graph, time_limit: float = 60.0, num_colors: int = None) -> dict:
    if len(graph) == 0:
        return {'coloring': {}, 'num_colors': 0, 'lower_bound': 0, 'status': 'optimal'}
    
    if num_colors is None:
        num_colors = max(dict(graph.degree()).values()) + 1
    
    vertices = list(graph.nodes())
    colors = range(num_colors)
    
    model = cp_model.CpModel()
    
    x = {(v, c): model.NewBoolVar(f'x_{v}_{c}') for v in vertices for c in colors}
    y = {c: model.NewBoolVar(f'y_{c}') for c in colors}
    
    for v in vertices:
        model.AddExactlyOne(x[v, c] for c in colors)
    
    for u, v in graph.edges():
        for c in colors:
            model.Add(x[u, c] + x[v, c] <= 1)
    
    for v in vertices:
        for c in colors:
            model.Add(x[v, c] <= y[c])
    
    model.Minimize(sum(y[c] for c in colors))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    
    result = {'coloring': None, 'num_colors': None, 'lower_bound': None, 'status': None}
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        result['status'] = 'optimal' if status == cp_model.OPTIMAL else 'feasible'
        result['coloring'] = {v: c for v in vertices for c in colors if solver.Value(x[v, c])}
        result['num_colors'] = sum(solver.Value(y[c]) for c in colors)
        result['lower_bound'] = int(solver.BestObjectiveBound())
    elif status == cp_model.INFEASIBLE:
        result['status'] = 'infeasible'
    else:
        result['status'] = 'timeout'
        if solver.BestObjectiveBound() is not None:
            result['lower_bound'] = int(solver.BestObjectiveBound())
    
    return result


def solve_ass_gurobi(graph: nx.Graph, time_limit: float = 60.0, num_colors: int = None) -> dict:
    if not GUROBI_AVAILABLE:
        raise ImportError("Gurobi is not installed")
    
    if len(graph) == 0:
        return {'coloring': {}, 'num_colors': 0, 'lower_bound': 0, 'status': 'optimal'}
    
    if num_colors is None:
        num_colors = max(dict(graph.degree()).values()) + 1
    
    vertices = list(graph.nodes())
    colors = range(num_colors)
    
    model = gp.Model("ass")
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    
    x = {(v, c): model.addVar(vtype=GRB.BINARY) for v in vertices for c in colors}
    y = {c: model.addVar(vtype=GRB.BINARY) for c in colors}
    
    for v in vertices:
        model.addConstr(gp.quicksum(x[v, c] for c in colors) == 1)
    
    for u, v in graph.edges():
        for c in colors:
            model.addConstr(x[u, c] + x[v, c] <= 1)
    
    for v in vertices:
        for c in colors:
            model.addConstr(x[v, c] <= y[c])
    
    model.setObjective(gp.quicksum(y[c] for c in colors), GRB.MINIMIZE)
    model.optimize()
    
    result = {'coloring': None, 'num_colors': None, 'lower_bound': None, 'status': None}
    
    if model.Status == GRB.OPTIMAL:
        result['status'] = 'optimal'
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        result['status'] = 'feasible'
    elif model.Status == GRB.INFEASIBLE:
        result['status'] = 'infeasible'
        return result
    else:
        result['status'] = 'timeout'
        if model.ObjBound is not None:
            result['lower_bound'] = int(model.ObjBound)
        return result
    
    result['coloring'] = {v: c for v in vertices for c in colors if x[v, c].X > 0.5}
    result['num_colors'] = int(round(model.ObjVal))
    result['lower_bound'] = int(model.ObjBound)
    
    return result
