from functools import wraps
from time import time

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        time_taken = te-ts
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, time_taken))
        return result, time_taken
    return wrap


def visualize_dag(W, ax=None, title="DAG", node_color='lightblue',
                 threshold=0.01, node_size=800, seed=42, pos=None):
    """
    Visualize a DAG from its weighted adjacency matrix.

    Parameters:
    -----------
    W : array-like
        Standard adjacency matrix where W[i,j] != 0 means edge from i→j
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure and axes are created
    title : str, optional
        Title for the plot
    node_color : str or list, optional
        Color(s) for the nodes
    threshold : float, optional
        Threshold for considering an edge present
    node_size : int, optional
        Size of the nodes in the visualization
    seed : int, optional
        Random seed for layout if pos is None
    pos : dict, optional
        Pre-computed positions for nodes. If None, positions are computed using spring layout

    Returns:
    --------
    tuple
        (NetworkX graph, node positions dict, matplotlib Axes)
    """
    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph()

    # Add nodes
    n_nodes = W.shape[0]
    G.add_nodes_from(range(n_nodes))

    # Convert to numpy if needed
    if hasattr(W, 'numpy'):
        W = W.numpy()

    # Add weighted edges where weight > threshold
    for i in range(n_nodes):
        for j in range(n_nodes):
            if abs(W[i, j]) > threshold:
                # i→j represents an edge from i to j (standard convention)
                G.add_edge(i, j, weight=float(W[i, j]))

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Compute layout if not provided
    if pos is None:
        pos = nx.spring_layout(G, seed=seed)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color,
                          alpha=0.8, ax=ax)

    # Draw edges with width proportional to weight
    edge_widths = [abs(G[u][v]['weight']) * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7,
                         arrowsize=25, arrowstyle='-|>', connectionstyle='arc3,rad=0.2',
                         ax=ax)

    # Draw node labels
    labels = {i: str(i) for i in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight='bold', ax=ax)

    # Optional: draw edge weights
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    # Set title and turn off axis
    ax.set_title(title, fontsize=15)
    ax.axis('off')

    return G, pos, ax


def plot_dag_comparison(W_true, W_est, figsize=(12, 5), threshold=0.01, transpose=True):
    """
    Plot the true and estimated DAGs side by side.

    Parameters:
    -----------
    W_true : array-like
        True weighted adjacency matrix
    W_est : array-like
        Estimated weighted adjacency matrix
    figsize : tuple, optional
        Figure size
    threshold : float, optional
        Threshold for considering an edge present
    transpose : bool, optional
        Whether to transpose the adjacency matrices (True for NOTEARS convention)

    Returns:
    --------
    tuple
        (figure, axes, graph positions dict)
    """
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot true DAG
    G_true, pos, _ = visualize_dag(W_true, ax=axes[0], title="True DAG",
                                 node_color='lightblue', threshold=threshold,
                                 )

    # Plot estimated DAG using same node positions
    G_est, _, _ = visualize_dag(W_est, ax=axes[1], title="Estimated DAG",
                              node_color='lightgreen', threshold=threshold,
                              pos=pos)

    plt.tight_layout()
    return fig, axes, pos


def evaluate_reconstruction(W_true, W_est, threshold=0.01, figsize=(12, 4),
                          show_plot=True, transpose=True):
    """
    Evaluate and visualize the reconstruction quality of the estimated DAG.

    Parameters:
    -----------
    W_true : array-like
        True weighted adjacency matrix
    W_est : array-like
        Estimated weighted adjacency matrix
    threshold : float, optional
        Threshold for considering an edge present
    figsize : tuple, optional
        Figure size for the plot
    show_plot : bool, optional
        Whether to show the plot
    transpose : bool, optional
        Whether to transpose the adjacency matrices (True for NOTEARS convention)

    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Ensure arrays are in numpy format
    W_true = np.array(W_true)
    W_est = np.array(W_est)

    # Transpose if using NOTEARS convention
    if transpose:
        W_true = W_true.T
        W_est = W_est.T

    # Threshold adjacency matrices
    true_edges = np.abs(W_true) > threshold
    est_edges = np.abs(W_est) > threshold

    # Compute evaluation metrics
    n_true_edges = np.sum(true_edges)
    n_est_edges = np.sum(est_edges)

    # True positive, false positive, false negative
    tp = np.sum((true_edges) & (est_edges))
    fp = np.sum((~true_edges) & (est_edges))
    fn = np.sum((true_edges) & (~est_edges))
    tn = np.sum((~true_edges) & (~est_edges))

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Compute SHD (Structural Hamming Distance)
    # Count edge additions, deletions, and reversals
    additions = np.sum((~true_edges) & (est_edges))
    deletions = np.sum((true_edges) & (~est_edges))

    # Identify edge reversals
    potential_reversals = np.zeros_like(true_edges)
    for i in range(W_true.shape[0]):
        for j in range(W_true.shape[0]):
            if true_edges[i, j] and est_edges[j, i]:
                potential_reversals[i, j] = 1

    # Avoid double-counting bidirectional edges as reversals
    reversals = np.sum(potential_reversals) - np.sum(potential_reversals & potential_reversals.T) / 2

    # SHD is the sum of additions, deletions and reversals
    shd = additions + deletions + reversals

    # Create result dictionary
    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_edges": n_true_edges,
        "est_edges": n_est_edges,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "shd": shd,
        "additions": additions,
        "deletions": deletions,
        "reversals": reversals
    }

    # Visualization
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Create binary matrices for visualization
        axes[0].imshow(true_edges.astype(int), cmap='Blues')
        axes[0].set_title("True DAG Structure")

        axes[1].imshow(est_edges.astype(int), cmap='Blues')
        axes[1].set_title("Estimated DAG Structure")

        # Create a difference matrix
        diff = np.zeros_like(true_edges, dtype=int)
        diff[true_edges & est_edges] = 3  # True positive
        diff[~true_edges & est_edges] = 2  # False positive
        diff[true_edges & ~est_edges] = 1  # False negative

        cmap = plt.cm.colors.ListedColormap(['white', 'red', 'green', 'blue'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        im = axes[2].imshow(diff, cmap=cmap, norm=norm)
        axes[2].set_title(f"Edge Recovery\nF1: {f1:.2f}, SHD: {shd}")

        # Add colorbar with labels
        cbar = fig.colorbar(im, ax=axes[2], ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['TN', 'FN', 'FP', 'TP'])

        plt.tight_layout()

    return results

