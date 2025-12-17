import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import ast
import pandas as pd
import io
import scipy

# --- Helper: Check for Pydot (Graphviz Export) ---
try:
    from networkx.drawing.nx_pydot import write_dot
    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

# --- Core Functions ---

def get_cycles_with_node_weight(G, source_node, target_val, weight_attr='weight'):
    """
    Optimized: Finds cycles passing through 'source_node' via an edge where 
    edge[weight_attr] == target_val.
    """
    # 1. Comparison Logic for Values (Handles floats vs strings)
    def check_match(val, target):
        try:
            return np.isclose(float(val), float(target))
        except:
            return str(val) == str(target)

    # 2. Identify Valid Neighbors (Next Steps)
    valid_neighbors = set()
    for u, v, data in G.edges(data=True):
        val = data.get(weight_attr, 1)
        
        if G.is_directed():
            if u == source_node:
                if check_match(val, target_val):
                    valid_neighbors.add(v)
        else:
            # Undirected: check both ways
            if u == source_node and check_match(val, target_val):
                valid_neighbors.add(v)
            elif v == source_node and check_match(val, target_val):
                valid_neighbors.add(u)

    if not valid_neighbors:
        return [], [], []

    # 3. Cycle Detection
    if G.is_directed():
        cycle_gen = nx.simple_cycles(G)
    else:
        cycle_gen = nx.cycle_basis(G)

    valid_cycles = []
    cycle_edges = set()
    
    # 4. Optimized Iteration
    for cycle in cycle_gen:
        if source_node not in cycle:
            continue

        idx = cycle.index(source_node)
        next_node = cycle[(idx + 1) % len(cycle)]
        
        is_valid_cycle = False
        if next_node in valid_neighbors:
            is_valid_cycle = True
        elif not G.is_directed():
            prev_node = cycle[(idx - 1) % len(cycle)]
            if prev_node in valid_neighbors:
                is_valid_cycle = True

        if is_valid_cycle:
            valid_cycles.append(cycle)
            for k in range(len(cycle)):
                u, v = cycle[k], cycle[(k + 1) % len(cycle)]
                cycle_edges.add((u, v))
                if not G.is_directed():
                    cycle_edges.add((v, u))

    target_edges = []
    for neighbor in valid_neighbors:
        target_edges.append((source_node, neighbor))
        if not G.is_directed():
            target_edges.append((neighbor, source_node))

    return valid_cycles, list(cycle_edges), target_edges

def matrix_to_graph(M, kind="auto"):
    """Converts a matrix M into a NetworkX graph."""
    M = np.array(M)
    M = np.nan_to_num(
