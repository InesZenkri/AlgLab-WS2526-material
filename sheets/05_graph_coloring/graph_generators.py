import networkx as nx


def generate_erdos_renyi():
    graphs = {}
    configs = [
        (50, 0.03, "er_50_ultra_sparse"),
        (50, 0.05, "er_50_v_sparse"),
        (50, 0.10, "er_50_sparse"),
        (50, 0.15, "er_50_mid"),
        (50, 0.20, "er_50_mid_dense"),
        (50, 0.30, "er_50_dense"),
        (100, 0.02, "er_100_ultra_sparse"),
        (100, 0.03, "er_100_v_sparse"),
        (100, 0.05, "er_100_sparse"),
        (100, 0.08, "er_100_mid"),
        (100, 0.12, "er_100_mid_dense"),
        (100, 0.20, "er_100_dense"),
        (200, 0.01, "er_200_ultra_sparse"),
        (200, 0.02, "er_200_v_sparse"),
        (200, 0.04, "er_200_sparse"),
        (200, 0.07, "er_200_mid"),
        (200, 0.10, "er_200_mid_dense"),
    ]
    for n, p, name in configs:
        graphs[name] = nx.erdos_renyi_graph(n=n, p=p, seed=42)
    return graphs


def generate_barabasi_albert():
    graphs = {}
    configs = [
        (50, 1, "ba_50_m1"), (50, 2, "ba_50_m2"), (50, 3, "ba_50_m3"), (50, 4, "ba_50_m4"),
        (100, 1, "ba_100_m1"), (100, 2, "ba_100_m2"), (100, 3, "ba_100_m3"), (100, 4, "ba_100_m4"), (100, 5, "ba_100_m5"),
        (200, 2, "ba_200_m2"), (200, 3, "ba_200_m3"), (200, 4, "ba_200_m4"),
    ]
    for n, m, name in configs:
        graphs[name] = nx.barabasi_albert_graph(n=n, m=m, seed=42)
    return graphs


def generate_kneser():
    graphs = {}
    configs = [(7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (9, 4), (10, 4), (11, 4)]
    for n, k in configs:
        graphs[f"kneser_{n}_{k}"] = nx.kneser_graph(n, k)
    return graphs


def generate_watts_strogatz():
    graphs = {}
    configs = [
        (50, 4, 0.05, "ws_50_k4_p005"), (50, 6, 0.05, "ws_50_k6_p005"),
        (50, 6, 0.10, "ws_50_k6_p01"), (50, 8, 0.30, "ws_50_k8_p03"),
        (100, 4, 0.02, "ws_100_k4_p002"), (100, 6, 0.05, "ws_100_k6_p005"),
        (100, 8, 0.10, "ws_100_k8_p01"), (100, 10, 0.30, "ws_100_k10_p03"),
        (200, 6, 0.05, "ws_200_k6_p005"), (200, 8, 0.10, "ws_200_k8_p01"),
    ]
    for n, k, p, name in configs:
        graphs[name] = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=42)
    return graphs


def generate_random_regular():
    graphs = {}
    configs = [
        (50, 3, "rr_50_d3"), (50, 4, "rr_50_d4"), (50, 6, "rr_50_d6"),
        (100, 3, "rr_100_d3"), (100, 4, "rr_100_d4"), (100, 6, "rr_100_d6"), (100, 8, "rr_100_d8"),
        (200, 4, "rr_200_d4"), (200, 6, "rr_200_d6"),
    ]
    for n, d, name in configs:
        graphs[name] = nx.random_regular_graph(d=d, n=n, seed=42)
    return graphs


def generate_all_graphs():
    graphs = {}
    graphs.update(generate_erdos_renyi())
    graphs.update(generate_barabasi_albert())
    graphs.update(generate_kneser())
    graphs.update(generate_watts_strogatz())
    graphs.update(generate_random_regular())
    return graphs


if __name__ == "__main__":
    graphs = generate_all_graphs()
    print(f"Total instances: {len(graphs)}\n")
    for name, G in graphs.items():
        print(f"{name:22s} | nodes={G.number_of_nodes():4d} | edges={G.number_of_edges():5d}")
