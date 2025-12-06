import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast
import pandas as pd  # Added for file handling

# --- Core Logic from Source (Unchanged) ---

def matrix_to_graph(M, kind="auto"):
    """
    Converts a matrix M into a NetworkX graph.
    """
    M = np.array(M)
    n_rows, n_cols = M.shape

    # --- Auto-detect representation type ---
    if kind == "auto":
        if n_rows == n_cols:
            kind = "adjacency"
        else:
            # Heuristic: if every column touches at most 2 vertices, treat as incidence 
            col_nnz = (M != 0).sum(axis=0)
            if np.all((col_nnz >= 1) & (col_nnz <= 2)):
                kind = "incidence"
            else:
                kind = "biadjacency"

    # --- Adjacency matrix case ---
    if kind == "adjacency":
        directed = not np.allclose(M, M.T)
        G = nx.from_numpy_array(M, create_using=nx.DiGraph if directed else nx.Graph)

    # --- Incidence matrix case (nodes x edges) ---
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

    # --- Bipartite (biadjacency) matrix case (U x V) ---
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

# --- Streamlit UI ---

st.set_page_config(page_title="Matrix to Graph Visualizer", layout="wide")

st.title("Interactive Graph Visualizer")
st.markdown("Enter a matrix below to visualize it as a network graph. The app supports **Adjacency**, **Incidence**, and **Biadjacency** matrices.")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# 1. File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Matrix (CSV/Excel)", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("--- OR ---")

# 2. Text Input (Fallback)
matrix_input = st.sidebar.text_area(
    "Input Matrix (Python List of Lists)",
    value="[[0, 1, 0], [1, 0, 1], [0, 1, 0]]",
    height=150,
    help="Example: [[0, 1], [1, 0]]"
)

kind_option = st.sidebar.selectbox(
    "Matrix Type",
    options=["auto", "adjacency", "incidence", "biadjacency"],
    index=0
)

# --- Processing Logic ---

if st.sidebar.button("Generate Graph"):
    try:
        matrix_data = None
        
        # Priority 1: Handle File Upload
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    # Assume no header for mathematical matrices
                    df = pd.read_csv(uploaded_file, header=None)
                else:
                    df = pd.read_excel(uploaded_file, header=None)
                
                # Convert DataFrame to List of Lists
                matrix_data = df.to_numpy().tolist()
                st.success(f"Loaded matrix from file: {uploaded_file.name}")
                
            except Exception as file_err:
                st.error(f"Error reading file: {file_err}")
                st.stop()

        # Priority 2: Handle Text Input
        elif matrix_input:
            matrix_data = ast.literal_eval(matrix_input)

        if matrix_data is None:
            st.error("No input provided.")
            st.stop()
        
        # --- Graph Generation ---
        G, detected_kind = matrix_to_graph(matrix_data, kind=kind_option)
        
        st.subheader(f"Result: {detected_kind.capitalize()} Matrix Detected")
        
        # Layout calculation
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Determine Layout based on detected kind 
            if detected_kind == "biadjacency":
                U_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
                pos = nx.bipartite_layout(G, U_nodes) 
            else:
                pos = nx.spring_layout(G, seed=42) 
            
            

            # Draw nodes and edges
            if isinstance(G, nx.DiGraph):
                nx.draw(G, pos, with_labels=True, node_size=800, arrows=True, ax=ax, node_color='lightblue')
            else:
                nx.draw(G, pos, with_labels=True, node_size=800, ax=ax, node_color='lightgreen')

            # Draw edge weights
            edge_labels = nx.get_edge_attributes(G, "weight")
            if edge_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
            
            st.pyplot(fig)

        with col2:
            st.info("Graph Statistics")
            st.write(f"**Nodes:** {G.number_of_nodes()}")
            st.write(f"**Edges:** {G.number_of_edges()}")
            st.write(f"**Directed:** {G.is_directed()}")
            if detected_kind == "biadjacency":
                 st.write("**Type:** Bipartite")
            
            # Show raw data preview if needed
            with st.expander("View Matrix Data"):
                st.write(np.array(matrix_data))

    except Exception as e:
        st.error(f"Error processing matrix: {e}")
        st.warning("If using text input, ensure it is a valid Python list of lists.")
