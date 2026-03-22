import networkx as nx

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from ortools.sat.python import cp_model


def solve_rep_cpsat(graph: nx.Graph, time_limit: float = 60.0) -> dict:
    if len(graph) == 0:
        return {'coloring': {}, 'num_colors': 0, 'lower_bound': 0, 'status': 'optimal'}
    
    vertices = list(graph.nodes())
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    neighbors = {v: set(graph.neighbors(v)) for v in vertices}
    
    model = cp_model.CpModel()
    
    x = {}
    for v in vertices:
        for w in vertices:
            if w not in neighbors[v] and vertex_to_idx[w] <= vertex_to_idx[v]:
                x[v, w] = model.NewBoolVar(f'x_{v}_{w}')
    
    for v in vertices:
        model.AddExactlyOne(x[v, w] for w in vertices if (v, w) in x)
    
    for u, v in graph.edges():
        for w in vertices:
            if (u, w) in x and (v, w) in x and (w, w) in x:
                model.Add(x[u, w] + x[v, w] <= x[w, w])
    
    model.Minimize(sum(x[v, v] for v in vertices if (v, v) in x))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    
    result = {'coloring': None, 'num_colors': None, 'lower_bound': None, 'status': None}
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        result['status'] = 'optimal' if status == cp_model.OPTIMAL else 'feasible'
        rep_to_color, color_counter = {}, 0
        result['coloring'] = {}
        for v in vertices:
            for w in vertices:
                if (v, w) in x and solver.Value(x[v, w]):
                    if w not in rep_to_color:
                        rep_to_color[w] = color_counter
                        color_counter += 1
                    result['coloring'][v] = rep_to_color[w]
                    break
        result['num_colors'] = color_counter
        result['lower_bound'] = int(solver.BestObjectiveBound())
    elif status == cp_model.INFEASIBLE:
        result['status'] = 'infeasible'
    else:
        result['status'] = 'timeout'
        if solver.BestObjectiveBound() is not None:
            result['lower_bound'] = int(solver.BestObjectiveBound())
    
    return result


def solve_rep_gurobi(graph: nx.Graph, time_limit: float = 60.0) -> dict:
    if not GUROBI_AVAILABLE:
        raise ImportError("Gurobi is not installed")
    
    if len(graph) == 0:
        return {'coloring': {}, 'num_colors': 0, 'lower_bound': 0, 'status': 'optimal'}
    
    vertices = list(graph.nodes())
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    neighbors = {v: set(graph.neighbors(v)) for v in vertices}
    
    model = gp.Model("rep")
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    
    x = {}
    for v in vertices:
        for w in vertices:
            if w not in neighbors[v] and vertex_to_idx[w] <= vertex_to_idx[v]:
                x[v, w] = model.addVar(vtype=GRB.BINARY)
    
    for v in vertices:
        model.addConstr(gp.quicksum(x[v, w] for w in vertices if (v, w) in x) == 1)
    
    for u, v in graph.edges():
        for w in vertices:
            if (u, w) in x and (v, w) in x and (w, w) in x:
                model.addConstr(x[u, w] + x[v, w] <= x[w, w])
    
    model.setObjective(gp.quicksum(x[v, v] for v in vertices if (v, v) in x), GRB.MINIMIZE)
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
    
    rep_to_color, color_counter = {}, 0
    result['coloring'] = {}
    for v in vertices:
        for w in vertices:
            if (v, w) in x and x[v, w].X > 0.5:
                if w not in rep_to_color:
                    rep_to_color[w] = color_counter
                    color_counter += 1
                result['coloring'][v] = rep_to_color[w]
                break
    result['num_colors'] = color_counter
    result['lower_bound'] = int(model.ObjBound)
    
    return result
