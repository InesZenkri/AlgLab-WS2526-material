import networkx as nx


class DegreeBasedPreprocessor:
    def __init__(self, graph: nx.Graph):
        self.original = graph
        self.removed_stack = []
    
    def preprocess(self, lower_bound: int = None) -> nx.Graph:
        if lower_bound is None:
            lower_bound = nx.approximation.large_clique_size(self.original)
        
        G = self.original.copy()
        self.removed_stack = []
        
        changed = True
        while changed:
            changed = False
            for v in list(G.nodes()):
                if G.degree(v) < lower_bound:
                    self.removed_stack.append((v, list(G.neighbors(v))))
                    G.remove_node(v)
                    changed = True
        
        return G
    
    def postprocess(self, coloring: dict, lower_bound: int) -> tuple[dict, int]:
        if coloring is None:
            return None, lower_bound
        
        result = dict(coloring)
        
        for v, neighbors in reversed(self.removed_stack):
            used = {result[n] for n in neighbors if n in result}
            c = 0
            while c in used:
                c += 1
            result[v] = c
        
        return result, lower_bound


if __name__ == "__main__":
    from heuristics.greedy import greedy_coloring
    
    G = nx.erdos_renyi_graph(50, 0.1, seed=42)
    print(f"Original: {G.number_of_nodes()}V, {G.number_of_edges()}E")
    
    prep = DegreeBasedPreprocessor(G)
    reduced = prep.preprocess()
    print(f"Reduced: {reduced.number_of_nodes()}V, removed {len(prep.removed_stack)}")
    
    coloring = greedy_coloring(reduced)
    full_coloring, _ = prep.postprocess(coloring, 2)
    print(f"Colors: {max(full_coloring.values()) + 1}")
