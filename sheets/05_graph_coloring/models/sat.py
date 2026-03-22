import networkx as nx
from pysat.solvers import Solver
from pysat.formula import CNF
import threading
import time


def can_color_with_k(graph: nx.Graph, k: int, timeout: float = None) -> tuple[str, dict | None]:
    if k <= 0:
        return ('unsat', None) if len(graph) > 0 else ('sat', {})
    
    vertices = list(graph.nodes())
    n = len(vertices)
    if n == 0:
        return ('sat', {})
    
    def var(v_idx, c):
        return v_idx * k + c + 1
    
    cnf = CNF()
    for v_idx in range(n):
        cnf.append([var(v_idx, c) for c in range(k)])
    
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    for u, v in graph.edges():
        for c in range(k):
            cnf.append([-var(vertex_to_idx[u], c), -var(vertex_to_idx[v], c)])
    
    with Solver(name='g3') as solver:
        solver.append_formula(cnf)
        
        timer = None
        if timeout is not None:
            timer = threading.Timer(timeout, solver.interrupt)
            timer.start()
        
        try:
            result = solver.solve_limited(expect_interrupt=True)
        finally:
            if timer:
                timer.cancel()
        
        if result is True:
            model = solver.get_model()
            coloring = {}
            for v_idx, v in enumerate(vertices):
                for c in range(k):
                    if var(v_idx, c) in model:
                        coloring[v] = c
                        break
            return ('sat', coloring)
        elif result is False:
            return ('unsat', None)
        else:
            return ('timeout', None)


def solve_sat(graph: nx.Graph, time_limit: float = 60.0) -> dict:
    if len(graph) == 0:
        return {'coloring': {}, 'num_colors': 0, 'lower_bound': 0, 'status': 'optimal'}
    
    start_time = time.time()
    lower_bound = nx.approximation.large_clique_size(graph)
    upper_bound = max(dict(graph.degree()).values()) + 1
    
    best_coloring, best_k, proven_unsat_at = None, None, None
    
    for k in range(upper_bound, lower_bound - 1, -1):
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break
        
        status, coloring = can_color_with_k(graph, k, timeout=time_limit - elapsed)
        
        if status == 'sat':
            best_coloring, best_k = coloring, k
        elif status == 'unsat':
            proven_unsat_at = k
            break
        else:
            break
    
    result = {'coloring': best_coloring, 'num_colors': best_k, 'lower_bound': lower_bound, 'status': 'timeout'}
    
    if best_coloring is not None:
        if (proven_unsat_at is not None and proven_unsat_at == best_k - 1) or best_k == lower_bound:
            result['status'] = 'optimal'
            result['lower_bound'] = best_k
        else:
            result['status'] = 'feasible'
    
    return result
