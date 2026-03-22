import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import time
from preprocessing import DegreeBasedPreprocessor
from models.cp_neq import solve_cp_neq

TIME_LIMIT = 60.0


def generate_instances():
    instances = {}
    seeds = [42, 123, 456]
    for n in [80, 120]:
        for p in [0.05, 0.1, 0.2]:
            for s in seeds:
                instances[f"ER_n{n}_p{p}_s{s}"] = nx.erdos_renyi_graph(n, p, seed=s)
    for n in [100, 150]:
        for m in [2, 4]:
            for s in seeds:
                instances[f"BA_n{n}_m{m}_s{s}"] = nx.barabasi_albert_graph(n, m, seed=s)
    for d in [3, 6]:
        for s in seeds:
            instances[f"RR_n100_d{d}_s{s}"] = nx.random_regular_graph(d, 100, seed=s)
    return instances


def run_benchmark():
    instances = generate_instances()
    results = []
    
    for name, G in instances.items():
        print(f"\n=== {name} ===")
        
        print("  No preprocessing...", end=" ", flush=True)
        t0 = time.time()
        r1 = solve_cp_neq(G, time_limit=TIME_LIMIT)
        t1 = time.time() - t0
        print(f"colors={r1['num_colors']}, time={t1:.2f}s")
        results.append({"instance": name, "strategy": "No Preprocessing", 
                       "colors": r1['num_colors'], "time": t1, "status": r1['status']})
        
        print("  With preprocessing...", end=" ", flush=True)
        t0 = time.time()
        prep = DegreeBasedPreprocessor(G)
        reduced = prep.preprocess()
        if reduced.number_of_nodes() > 0:
            r2 = solve_cp_neq(reduced, time_limit=TIME_LIMIT)
            coloring, lb = prep.postprocess(r2['coloring'], r2['lower_bound'])
        else:
            coloring, lb = prep.postprocess({}, 1)
            r2 = {'status': 'optimal'}
        t2 = time.time() - t0
        num_colors = max(coloring.values()) + 1 if coloring else 0
        print(f"colors={num_colors}, time={t2:.2f}s (reduced {G.number_of_nodes()}->{reduced.number_of_nodes()})")
        results.append({"instance": name, "strategy": "With Preprocessing",
                       "colors": num_colors, "time": t2, "status": r2['status']})
    
    return pd.DataFrame(results)


def plot_performance_profile(
    data: pd.DataFrame, instance_column: str, strategy_column: str,
    metric_column: str, direction: str, title: str = None,
    highlight_best: bool = False, ax: Axes = None, figsize: tuple = (9, 6),
    scale: str = None, log_base: int = 2
) -> Axes:
    best_val = data.groupby(instance_column)[metric_column].agg(direction)
    pivot = data.groupby([instance_column, strategy_column])[metric_column].median().unstack(fill_value=np.nan)
    
    comp = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for strat in pivot.columns:
        comp[strat] = pivot[strat] / best_val if direction == "min" else best_val / pivot[strat]
    comp = comp.replace([np.inf, -np.inf, 0.0], np.nan)
    
    all_vals = comp.values.flatten()
    finite_vals = all_vals[np.isfinite(all_vals)]
    all_x = np.unique(np.concatenate(([1.0], np.sort(finite_vals))))
    
    n_instances = comp.shape[0]
    profile = pd.DataFrame(index=all_x, columns=comp.columns, dtype=float)
    for x in all_x:
        profile.loc[x] = (comp <= x).sum(axis=0) / n_instances
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for strat in profile.columns:
        ax.step(all_x, profile[strat].astype(float), where="post", label=strat, linewidth=2)
    
    # Use log scale if range is large or explicitly requested
    use_log = scale == "log" or (scale is None and all_x[-1] > 2)
    if use_log:
        ax.set_xscale("log", base=log_base)
        ax.set_xlim(1.0, all_x[-1] * 1.1)
        xlabel = f"Within this factor of the best (log{log_base} scale)"
    else:
        ax.set_xlim(1.0, max(1.5, all_x[-1] * 1.1))
        xlabel = "Within this factor of the best"
    
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proportion of problems")
    ax.set_title(title or "Performance Profile")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    return ax


if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESSING BENCHMARK")
    print("=" * 60)
    
    df = run_benchmark()
    df.to_csv("preprocessing_results.csv", index=False)
    
    ax = plot_performance_profile(
        df, instance_column="instance", strategy_column="strategy",
        metric_column="time", direction="min",
        title="Preprocessing Impact: Solving Time", highlight_best=True
    )
    ax.figure.savefig("preprocessing_profile.png", dpi=150)
    print("\nSaved: preprocessing_profile.png")
