import networkx as nx
import matplotlib.pyplot as plt
from preprocessing import DegreeBasedPreprocessor


def plot_preprocessing_comparison(graphs: dict, output_file: str = "preprocessing_comparison.png"):
    n = len(graphs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    
    if n == 1:
        axes = [axes]
    
    for i, (name, G) in enumerate(graphs.items()):
        prep = DegreeBasedPreprocessor(G)
        reduced = prep.preprocess()
        
        # Original
        ax1 = axes[i][0]
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax1, node_size=30, node_color='steelblue', 
                edge_color='gray', alpha=0.7, with_labels=False)
        ax1.set_title(f"{name} - Original\n{G.number_of_nodes()}V, {G.number_of_edges()}E")
        
        # Preprocessed
        ax2 = axes[i][1]
        if reduced.number_of_nodes() > 0:
            pos2 = nx.spring_layout(reduced, seed=42)
            nx.draw(reduced, pos2, ax=ax2, node_size=30, node_color='coral',
                    edge_color='gray', alpha=0.7, with_labels=False)
        ax2.set_title(f"{name} - After Preprocessing\n{reduced.number_of_nodes()}V, {reduced.number_of_edges()}E\n(removed {len(prep.removed_stack)})")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    graphs = {
        "ER_n50_p0.1": nx.erdos_renyi_graph(50, 0.1, seed=42),
        "BA_n50_m2": nx.barabasi_albert_graph(50, 2, seed=42),
        "RR_n50_d4": nx.random_regular_graph(4, 50, seed=42),
        "WS_n50_k4_p0.1": nx.watts_strogatz_graph(50, 4, 0.1, seed=42),
    }
    
    plot_preprocessing_comparison(graphs)
