import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import time

from preprocessing import DegreeBasedPreprocessor
from models.cp_neq import solve_cp_neq
from models.ass import solve_ass_cpsat
from models.ass_symmetry import solve_ass_symmetry_cpsat
from models.cp_alldiff import solve_cp_alldiff
from models.sat import solve_sat

TIME_LIMIT = 60.0

SOLVERS = {
    "CP-NEQ": solve_cp_neq,
    "ASS-CPSAT": solve_ass_cpsat,
    "ASS-S-CPSAT": solve_ass_symmetry_cpsat,
    "CP-AllDiff": solve_cp_alldiff,
    "SAT": solve_sat,
}


def generate_instances():
    instances = {}
    seeds = [42, 123]
    for n in [80, 120]:
        for p in [0.05, 0.1, 0.2]:
            for s in seeds:
                instances[f"ER_n{n}_p{p}_s{s}"] = nx.erdos_renyi_graph(n, p, seed=s)
    for n in [100, 150]:
        for m in [2, 4]:
            for s in seeds:
                instances[f"BA_n{n}_m{m}_s{s}"] = nx.barabasi_albert_graph(n, m, seed=s)
    return instances


def run_benchmark():
    instances = generate_instances()
    results = []
    
    for name, G in instances.items():
        print(f"\n=== {name} ===")
        prep = DegreeBasedPreprocessor(G)
        reduced = prep.preprocess()
        
        for solver_name, solver_func in SOLVERS.items():
            print(f"  {solver_name}...", end=" ", flush=True)
            t0 = time.time()
            r1 = solver_func(G, time_limit=TIME_LIMIT)
            t1 = time.time() - t0
            results.append({"instance": name, "solver": f"{solver_name}", "preprocessing": "No",
                           "colors": r1['num_colors'], "lower_bound": r1['lower_bound'], 
                           "time": t1, "status": r1['status']})
            
            t0 = time.time()
            if reduced.number_of_nodes() > 0:
                r2 = solver_func(reduced, time_limit=TIME_LIMIT)
                coloring, lb = prep.postprocess(r2['coloring'], r2['lower_bound'])
                num_colors = max(coloring.values()) + 1 if coloring else 0
            else:
                coloring, lb = prep.postprocess({}, 1)
                num_colors = max(coloring.values()) + 1 if coloring else 0
                r2 = {'status': 'optimal', 'lower_bound': lb}
            t2 = time.time() - t0
            results.append({"instance": name, "solver": f"{solver_name}", "preprocessing": "Yes",
                           "colors": num_colors, "lower_bound": r2['lower_bound'],
                           "time": t2, "status": r2['status']})
            print(f"done ({t1:.1f}s / {t2:.1f}s)")
    
    return pd.DataFrame(results)


def plot_performance_profile(data, instance_column, strategy_column, metric_column, 
                            direction, title=None, scale=None, log_base=2, figsize=(10, 6)):
    best_val = data.groupby(instance_column)[metric_column].agg(direction)
    pivot = data.groupby([instance_column, strategy_column])[metric_column].median().unstack(fill_value=np.nan)
    
    comp = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for strat in pivot.columns:
        comp[strat] = pivot[strat] / best_val if direction == "min" else best_val / pivot[strat]
    comp = comp.replace([np.inf, -np.inf, 0.0], np.nan)
    
    all_x = np.unique(np.concatenate(([1.0], np.sort(comp.values.flatten()[np.isfinite(comp.values.flatten())]))))
    
    n_instances = comp.shape[0]
    profile = pd.DataFrame(index=all_x, columns=comp.columns, dtype=float)
    for x in all_x:
        profile.loc[x] = (comp <= x).sum(axis=0) / n_instances
    
    fig, ax = plt.subplots(figsize=figsize)
    for strat in profile.columns:
        ax.step(all_x, profile[strat].astype(float), where="post", label=strat, linewidth=1.5)
    
    use_log = scale == "log" or (scale is None and all_x[-1] > 2)
    if use_log:
        ax.set_xscale("log", base=log_base)
    ax.set_xlim(1.0, max(1.5, all_x[-1] * 1.1))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(f"Within this factor of the best{' (log2)' if use_log else ''}")
    ax.set_ylabel("Proportion of problems")
    ax.set_title(title or "Performance Profile")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return ax


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK: ALL MODELS WITH PREPROCESSING")
    print("=" * 60)
    
    df = run_benchmark()
    df.to_csv("benchmark_with_preprocessing.csv", index=False)
    
    df['strategy'] = df['solver'] + ' (' + df['preprocessing'] + ' prep)'
    
    ax = plot_performance_profile(df, "instance", "strategy", "colors", "min",
                                  title="Solution Quality: All Models (With/Without Preprocessing)")
    ax.figure.savefig("profile_colors_preprocessing.png", dpi=150)
    print("\nSaved: profile_colors_preprocessing.png")
    plt.close()
    
    ax = plot_performance_profile(df, "instance", "strategy", "time", "min",
                                  title="Solving Time: All Models (With/Without Preprocessing)")
    ax.figure.savefig("profile_time_preprocessing.png", dpi=150)
    print("Saved: profile_time_preprocessing.png")
    plt.close()
    
    ax = plot_performance_profile(df, "instance", "preprocessing", "time", "min",
                                  title="Preprocessing Impact on Solving Time")
    ax.figure.savefig("profile_preprocessing_time.png", dpi=150)
    print("Saved: profile_preprocessing_time.png")
    plt.close()
    
    print("\nDone!")
