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
    # We only care about cycles where source_node -> neighbor has specific value
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

    # 3. Cycle Detection (Generator based for performance)
    if G.is_directed():
        cycle_gen = nx.simple_cycles(G) # Generator
    else:
        cycle_gen = nx.cycle_basis(G)   # List (fast for undirected)

    valid_cycles = []
    cycle_edges = set()
    
    # 4. Optimized Iteration
    for cycle in cycle_gen:
        if source_node not in cycle:
            continue

        # Check if cycle uses a valid connection from source_node
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
            # Add all edges of this cycle to set
            for k in range(len(cycle)):
                u, v = cycle[k], cycle[(k + 1) % len(cycle)]
                cycle_edges.add((u, v))
                if not G.is_directed():
                    cycle_edges.add((v, u))

    # Reconstruct target edges list for visualization
    target_edges = []
    for neighbor in valid_neighbors:
        target_edges.append((source_node, neighbor))
        if not G.is_directed():
            target_edges.append((neighbor, source_node))

    return valid_cycles, list(cycle_edges), target_edges

def matrix_to_graph(M, kind="auto"):
    """Converts a matrix M into a NetworkX graph."""
    M = np.array(M)
    M = np.nan_to_num(M) 
    n_rows, n_cols = M.shape

    if kind == "auto":
        if n_rows == n_cols:
            kind = "adjacency"
        else:
            col_nnz = (M != 0).sum(axis=0)
            if np.all((col_nnz >= 1) & (col_nnz <= 2)):
                kind = "incidence"
            else:
                kind = "biadjacency"

    if kind == "adjacency":
        directed = not np.allclose(M, M.T)
        G = nx.from_numpy_array(M, create_using=nx.DiGraph if directed else nx.Graph)

    elif kind == "incidence":
        directed = np.any(M < 0)
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(n_rows))
        for e in range(n_cols):
            col = M[:, e]
            if directed:
                tails = np.where(col > 0)[0]
                heads = np.where(col < 0)[0]
                for t in tails:
                    for h in heads:
                        G.add_edge(int(t), int(h), weight=abs(col[t]))
            else:
                nodes = np.where(col != 0)[0]
                if len(nodes) == 2:
                    u, v = nodes
                    G.add_edge(int(u), int(v), weight=col[u])

    elif kind == "biadjacency":
        G = nx.Graph()
        U_nodes = [f"u{i}" for i in range(n_rows)]
        V_nodes = [f"v{j}" for j in range(n_cols)]
        G.add_nodes_from(U_nodes, bipartite=0)
        G.add_nodes_from(V_nodes, bipartite=1)
        for i in range(n_rows):
            for j in range(n_cols):
                if M[i, j] != 0:
                    G.add_edge(U_nodes[i], V_nodes[j], weight=M[i, j])
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return G, kind

def load_matrix_smart(uploaded_file):
    """Robustly loads a matrix from CSV/Excel."""
    uploaded_file.seek(0)
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    def read_func(file, **kwargs):
        if 'csv' in file_type:
            return pd.read_csv(file, **kwargs)
        else:
            return pd.read_excel(file, **kwargs)

    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, header=None)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all(): return df_numeric.values.tolist()
    except: pass

    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, index_col=0)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all() and df_numeric.shape[0] > 0: return df_numeric.values.tolist()
    except: pass
    
    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, header=0)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all(): return df_numeric.values.tolist()
    except: pass

    raise ValueError("Could not parse file as a numeric matrix.")

# --- UI Setup ---

st.set_page_config(page_title="Graph Viz Pro", layout="wide")
st.title("Interactive Graph Visualizer")

# --- Sidebar ---
st.sidebar.header("1. Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
st.sidebar.markdown("**OR**")
matrix_input = st.sidebar.text_area("Paste Python List", value="[[0, 1, 0], [1, 0, 1], [0, 1, 0]]", height=100)

st.sidebar.header("2. Logic")
kind_option = st.sidebar.selectbox("Matrix Type", options=["auto", "adjacency", "incidence", "biadjacency"])

# --- Main Logic ---

matrix_data = None
try:
    if uploaded_file:
        matrix_data = load_matrix_smart(uploaded_file)
        st.sidebar.success(f"Loaded {len(matrix_data)}x{len(matrix_data[0])} matrix.")
    elif matrix_input:
        matrix_data = ast.literal_eval(matrix_input)
except Exception as e:
    st.error(f"
