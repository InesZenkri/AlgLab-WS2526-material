import networkx as nx
import pandas as pd


def validate_coloring(graph: nx.Graph, coloring: dict) -> tuple[bool, str]:
    if coloring is None:
        return False, "No coloring"
    
    missing = set(graph.nodes()) - set(coloring.keys())
    if missing:
        return False, f"Missing {len(missing)} vertices"
    
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            return False, f"Conflict: {u}-{v} both color {coloring[u]}"
    
    return True, "Valid"


def validate_benchmark_results(csv_path: str = "benchmark_results.csv"):
    from benchmark_models import generate_instances
    
    df = pd.read_csv(csv_path)
    instances = generate_instances()
    
    print("Checking for inconsistencies in benchmark results...\n")
    
    issues = []
    
    for inst_name in df['instance'].unique():
        inst_df = df[df['instance'] == inst_name]
        colors = inst_df['colors'].dropna()
        
        if len(colors) == 0:
            continue
        
        best = colors.min()
        worst = colors.max()
        
        if worst > best * 3:
            bad = inst_df[inst_df['colors'] > best * 3][['solver', 'colors']]
            for _, row in bad.iterrows():
                issues.append(f"{inst_name}: {row['solver']} found {int(row['colors'])} colors (best={int(best)})")
    
    if issues:
        print(f"Found {len(issues)} suspicious results:\n")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("No obvious inconsistencies found.")
    
    print(f"\n\nSummary:")
    print(f"  Total results: {len(df)}")
    print(f"  Optimal: {(df['status'] == 'optimal').sum()}")
    print(f"  Feasible: {(df['status'] == 'feasible').sum()}")
    print(f"  Timeout: {(df['status'] == 'timeout').sum()}")


if __name__ == "__main__":
    validate_benchmark_results()
