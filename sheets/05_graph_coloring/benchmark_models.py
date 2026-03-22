import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.cp_neq import solve_cp_neq
from models.ass import solve_ass_cpsat, solve_ass_gurobi, GUROBI_AVAILABLE
from models.ass_symmetry import solve_ass_symmetry_cpsat, solve_ass_symmetry_gurobi
from models.rep import solve_rep_cpsat, solve_rep_gurobi
from models.cp_alldiff import solve_cp_alldiff
from models.sat import solve_sat

TIME_LIMIT = 60.0

SOLVERS = {
    "CP-NEQ": solve_cp_neq,
    "ASS-CPSAT": solve_ass_cpsat,
    "ASS-S-CPSAT": solve_ass_symmetry_cpsat,
    "REP-CPSAT": solve_rep_cpsat,
    "CP-AllDiff": solve_cp_alldiff,
    "SAT": solve_sat,
}

if GUROBI_AVAILABLE:
    SOLVERS["ASS-Gurobi"] = solve_ass_gurobi
    SOLVERS["ASS-S-Gurobi"] = solve_ass_symmetry_gurobi
    SOLVERS["REP-Gurobi"] = solve_rep_gurobi


def generate_instances():
    instances = {}
    seeds = [42, 123, 456]
    
    # Erdos-Renyi (random density sweep)
    for n in [80, 120]:
        for p in [0.05, 0.10, 0.20, 0.35]:
            for s in seeds:
                instances[f"ER_n{n}_p{p}_s{s}"] = nx.erdos_renyi_graph(n, p, seed=s)
    
    # Barabasi-Albert (scale-free hubs)
    for n in [100, 150]:
        for m in [2, 4, 6]:
            for s in seeds:
                instances[f"BA_n{n}_m{m}_s{s}"] = nx.barabasi_albert_graph(n, m, seed=s)
    
    # Watts-Strogatz (small-world)
    for k in [6, 10]:
        for p in [0.01, 0.10, 0.30]:
            for s in seeds:
                instances[f"WS_n100_k{k}_p{p}_s{s}"] = nx.watts_strogatz_graph(100, k, p, seed=s)
    
    # Random regular (uniform degree)
    for n in [100, 150]:
        for d in [3, 6, 10]:
            for s in seeds:
                instances[f"RR_n{n}_d{d}_s{s}"] = nx.random_regular_graph(d, n, seed=s)
    
    # Kneser graphs (structured/chromatic traps)
    for n, k in [(9,3), (10,3), (11,3), (9,4), (10,4), (11,4)]:
        instances[f"Kneser_{n}_{k}"] = nx.kneser_graph(n, k)
    
    instances["Petersen"] = nx.petersen_graph()
    instances["Cycle_50"] = nx.cycle_graph(50)
    instances["Wheel_20"] = nx.wheel_graph(20)
    instances["Complete_8"] = nx.complete_graph(8)
    
    return instances


def run_benchmark():
    instances = generate_instances()
    results = []
    
    for inst_name, graph in instances.items():
        print(f"\n=== {inst_name} ({graph.number_of_nodes()}V, {graph.number_of_edges()}E) ===")
        
        for solver_name, solver_func in SOLVERS.items():
            print(f"  {solver_name}...", end=" ", flush=True)
            
            try:
                result = solver_func(graph, time_limit=TIME_LIMIT)
                num_colors = result.get('num_colors')
                lower_bound = result.get('lower_bound')
                status = result.get('status')
                print(f"colors={num_colors}, lb={lower_bound}, status={status}")
            except Exception as e:
                print(f"ERROR: {e}")
                num_colors, lower_bound, status = None, None, 'error'
            
            results.append({
                "instance": inst_name,
                "solver": solver_name,
                "colors": num_colors,
                "lower_bound": lower_bound,
                "status": status
            })
    
    return pd.DataFrame(results)


from matplotlib.axes import Axes


def plot_performance_profile(
    data: pd.DataFrame,
    instance_column: str,
    strategy_column: str,
    metric_column: str,
    direction: str,
    comparison: str = "relative",
    title: str | None = None,
    highlight_best: bool = False,
    ax: Axes | None = None,
    scale: str | None = None,
    log_base: int = 2,
    figsize: tuple = (9, 6),
) -> Axes:
    if direction not in ("min", "max"):
        raise ValueError("`direction` must be 'min' or 'max'.")
    if comparison not in ("relative", "absolute"):
        raise ValueError("`comparison` must be 'relative' or 'absolute'.")

    best_val = data.groupby(instance_column)[metric_column].agg(direction)

    pivot = (
        data.groupby([instance_column, strategy_column])[metric_column]
        .median()
        .unstack(fill_value=np.nan)
    )

    comp = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)

    if comparison == "relative":
        for strat in pivot.columns:
            if direction == "min":
                comp[strat] = pivot[strat] / best_val
            else:
                comp[strat] = best_val / pivot[strat]
        comp = comp.replace([np.inf, -np.inf, 0.0], np.nan)
    else:
        for strat in pivot.columns:
            if direction == "min":
                comp[strat] = pivot[strat] - best_val
            else:
                comp[strat] = best_val - pivot[strat]
        comp = comp.replace([np.inf, -np.inf], np.nan)

    all_vals = comp.values.flatten()
    finite_vals = all_vals[np.isfinite(all_vals)]
    baseline = 1.0 if comparison == "relative" else 0.0
    all_x = np.unique(np.sort(finite_vals))
    all_x = np.concatenate(([baseline], all_x))
    all_x = np.unique(np.sort(all_x))

    n_instances = comp.shape[0]
    profile = pd.DataFrame(index=all_x, columns=comp.columns, dtype=float)

    for x in all_x:
        leq = (comp <= x).sum(axis=0)
        profile.loc[x] = leq / n_instances

    best_solver = None
    if highlight_best:
        if comparison == "relative":
            log_x = np.log(all_x)
            areas = {}
            for strat in profile.columns:
                y = profile[strat].astype(float).values
                areas[strat] = np.trapezoid(y, x=log_x)
            best_solver = max(areas, key=areas.get)
        else:
            areas = {}
            for strat in profile.columns:
                y = profile[strat].astype(float).values
                areas[strat] = np.trapezoid(y, x=all_x)
            best_solver = max(areas, key=areas.get)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if scale is None:
        if comparison == "relative" and all_x[-1] > 10:
            use_log = True
        else:
            use_log = False
    else:
        use_log = scale == "log"

    for strat in profile.columns:
        y = profile[strat].astype(float)
        if highlight_best and strat == best_solver:
            ax.step(all_x, y, where="post", label=strat, linewidth=3.0, alpha=1.0)
        else:
            ax.step(all_x, y, where="post", label=strat, linewidth=1.5,
                    alpha=0.6 if highlight_best else 1.0)

    if comparison == "relative":
        if use_log:
            ax.set_xscale("log", base=log_base)
            ax.set_xlim(all_x[1], all_x[-1] * 1.1)
        else:
            ax.set_xscale("linear")
            ax.set_xlim(1.0, all_x[-1] * 1.1)
        xlabel = (f"Within this factor of the best (log{log_base} scale)"
                  if use_log else "Within this factor of the best (linear scale)")
    else:
        ax.set_xscale("linear")
        ax.set_xlim(0.0, all_x[-1] * 1.1)
        xlabel = "Absolute difference from the best"

    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Proportion of problems", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, pad=14)
    else:
        ax.set_title("Performance Profile", fontsize=14, pad=14)

    ax.axvline(x=baseline, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    return ax


def print_summary(df):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    pivot = df.pivot(index="instance", columns="solver", values="colors")
    print("\nColors found per instance:")
    print(pivot.to_string())
    
    print("\n\nNumber of instances where each solver found the best solution:")
    best_per_instance = pivot.min(axis=1)
    for solver in pivot.columns:
        wins = (pivot[solver] == best_per_instance).sum()
        print(f"  {solver}: {wins}")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK 5: BENCHMARKING ALL MODELS")
    print("=" * 60)
    
    df = run_benchmark()
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")
    
    # Performance profile for solution quality (colors - lower is better)
    ax = plot_performance_profile(
        df,
        instance_column="instance",
        strategy_column="solver",
        metric_column="colors",
        direction="min",
        title="Performance Profile: Solution Quality",
        highlight_best=True,
    )
    ax.figure.savefig("profile_colors.png", dpi=150)
    print("Saved: profile_colors.png")
    plt.close(ax.figure)
    
    # Performance profile for lower bounds (higher is better)
    ax = plot_performance_profile(
        df,
        instance_column="instance",
        strategy_column="solver",
        metric_column="lower_bound",
        direction="max",
        title="Performance Profile: Lower Bounds",
        highlight_best=True,
    )
    ax.figure.savefig("profile_bounds.png", dpi=150)
    print("Saved: profile_bounds.png")
    plt.close(ax.figure)
    
    print_summary(df)
    print("\nDone!")
