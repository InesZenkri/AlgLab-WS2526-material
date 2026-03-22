# MIT License
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from graph_generators import generate_all_graphs
from heuristics import greedy_coloring, multistart_greedy, dsatur, num_colors


def benchmark_heuristics():
    """Run all heuristics on all graphs and collect results."""
    graphs = generate_all_graphs()
    results = []
    
    for name, G in graphs.items():
        # Naive greedy
        coloring = greedy_coloring(G)
        results.append({"instance": name, "strategy": "greedy", "colors": num_colors(coloring)})
        
        # Multi-start greedy
        coloring = multistart_greedy(G, iterations=50, seed=42)
        results.append({"instance": name, "strategy": "multistart", "colors": num_colors(coloring)})
        
        # DSATUR
        coloring = dsatur(G)
        results.append({"instance": name, "strategy": "dsatur", "colors": num_colors(coloring)})
    
    return pd.DataFrame(results)


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
    """
    Plot a performance profile (from CP-SAT Primer).
    """
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
                areas[strat] = np.trapz(y, x=log_x)
            best_solver = max(areas, key=areas.get)
        else:
            areas = {}
            for strat in profile.columns:
                y = profile[strat].astype(float).values
                areas[strat] = np.trapz(y, x=all_x)
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
            ax.step(
                all_x, y, where="post", label=strat, linewidth=1.5,
                alpha=0.6 if highlight_best else 1.0,
            )

    if comparison == "relative":
        if use_log:
            ax.set_xscale("log", base=log_base)
            ax.set_xlim(all_x[1], all_x[-1] * 1.1)
        else:
            ax.set_xscale("linear")
            ax.set_xlim(1.0, all_x[-1] * 1.1)
        xlabel = (
            f"Within this factor of the best (log{log_base} scale)"
            if use_log
            else "Within this factor of the best (linear scale)"
        )
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


if __name__ == "__main__":
    print("Running heuristics benchmark...")
    df = benchmark_heuristics()
    
    print("\nResults:")
    print(df.pivot(index="instance", columns="strategy", values="colors").to_string())
    
    print("\nGenerating performance profile...")
    ax = plot_performance_profile(
        df,
        instance_column="instance",
        strategy_column="strategy",
        metric_column="colors",
        direction="min",
        title="Heuristics Performance Profile",
        highlight_best=True,
    )
    plt.savefig("heuristics_performance.png", dpi=150)
    plt.show()
