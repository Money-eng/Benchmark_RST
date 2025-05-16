#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_filtration.py
-------------------
Utilities to compute geodesic (shortest-path) distance-based filtrations on graphs
for TDA pipelines (e.g., GUDHI, giotto-tda).

One filtration is implemented:

- graph_geodesic_filtration: based on shortest-path distances on an arbitrary graph.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import networkx as nx

__all__ = ["graph_geodesic_filtration"]

logger = logging.getLogger(__name__)


def graph_geodesic_filtration(
    mask: Union[np.ndarray, list],
    G: nx.Graph,
    seed: Any,
    weight: Optional[str] = None,
    inverse: bool = True,
    nodes: Optional[List[Any]] = None,
    visualize: bool = False
) -> Tuple[np.ndarray, List[Any]]:
    """
    Compute a geodesic (shortest-path) distance-based filtration on a graph.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph whose nodes are any hashable types.
    seed : Any
        The node from which to compute distances. Must be in G.
    weight : str or None, default=None
        Edge attribute key to use as weight; if None, treats all edges as unit weight.
    inverse : bool, default=True
        If True, returns max_distance - distance so that nodes closer to the seed
        appear earlier (lower values) in sub-level set filtrations.
    nodes : list[Any] or None, default=None
        The ordering of nodes for the returned array. If None, uses list(G.nodes()).
    visualize : bool, default=False
        If True, draws the graph with nodes colored by filtration values.

    Returns
    -------
    filtration : np.ndarray of float32, shape (n_nodes,)
        Filtration values for each node in the order given by `nodes`.
    nodes : list[Any]
        The list of nodes corresponding to the entries in `filtration`.
    """
    if seed not in G:
        logger.error("Seed node %r not found in graph.", seed)
        raise KeyError(f"Seed node {seed} not in graph.")

    # Determine node order
    if nodes is None:
        nodes = list(G.nodes())

    # Compute shortest-path lengths
    try:
        if weight:
            lengths = nx.single_source_dijkstra_path_length(G, seed, weight=weight)
        else:
            lengths = nx.single_source_shortest_path_length(G, seed)
    except Exception as e:
        logger.exception("Error computing shortest-path distances: %s", e)
        raise

    # Build distances array
    dist_arr = np.array([lengths.get(n, np.inf) for n in nodes], dtype=np.float32)
    finite = np.isfinite(dist_arr)
    if not finite.any():
        logger.warning("No nodes reachable from seed %r; returning zeros.", seed)
        filt = np.zeros_like(dist_arr)
    else:
        maxd = float(np.max(dist_arr[finite]))
        filt = (maxd - dist_arr) if inverse else dist_arr
        # Unreachable nodes have distance=inf → filtration = maxd - inf = -inf; clip
        filt = np.where(np.isfinite(filt), filt, 0.0)

    # Optional visualization
    if visualize:
        try:
            import matplotlib.pyplot as plt
            pos = nx.spring_layout(G)
            plt.figure(figsize=(8, 6))
            nx.draw(
                G, pos,
                node_color=filt,
                cmap='hot',
                with_labels=True,
                node_size=300,
                edge_color='gray',
                linewidths=0.5
            )
            sm = plt.cm.ScalarMappable(
                cmap='hot',
                norm=plt.Normalize(vmin=filt.min(), vmax=filt.max())
            )
            plt.colorbar(sm, label='Filtration value')
            plt.title('Graph Geodesic Filtration')
            plt.axis('off')
            plt.show()
        except ImportError:
            logger.warning("matplotlib is required for visualization.")

    return filt.astype(np.float32), nodes
