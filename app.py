import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast
import pandas as pd
import io

# --- Core Logic (Unchanged) ---

def matrix_to_graph(M, kind="auto"):
    """
    Converts a matrix M into a NetworkX graph.
    """
    M = np.array(M)
    # Ensure numeric type, replacing any residual NaNs with 0
    M = np.nan_to_num(M) 
    n_rows, n_cols = M.shape

    # --- Auto-detect representation type ---
    if kind == "auto":
        if n_rows == n_cols:
            kind = "adjacency"
        else:
            col_nnz = (M != 0).sum(axis=0)
            if np.all((col_nnz >= 1) & (col_nnz <= 2)):
                kind = "incidence"
            else:
                kind = "biadjacency"

    # --- Adjacency matrix case ---
    if kind == "adjacency":
        directed = not np.allclose(M, M.T)
        G = nx.from_numpy_array(M, create_using=nx.DiGraph if directed else nx.Graph)

    # --- Incidence matrix case ---
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

    # --- Bipartite matrix case ---
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
    """
    Robustly loads a matrix from CSV/Excel, handling headers and indices.
    """
    uploaded_file.seek(0)
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # 1. Helper to read based on type
    def read_func(file, **kwargs):
        if 'csv' in file_type:
            return pd.read_csv(file, **kwargs)
        else:
            return pd.read_excel(file, **kwargs)

    # 2. Strategy A: Try reading as raw matrix (no headers)
    # This works for [[0,1],[1,0]] files
    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, header=None)
        # Check if strictly numeric
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all():
            return df_numeric.values.tolist()
    except:
        pass

    # 3. Strategy B: Try reading with Header and Index (Common for exported DataFrames)
    # This works for the file you uploaded (with index column)
    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, index_col=0)
        # Ensure body is numeric
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all() and df_numeric.shape[0] > 0:
            return df_numeric.values.tolist()
    except:
        pass
    
    # 4. Strategy C: Try reading with Header but NO Index
    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, header=0)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all():
            return df_numeric.values.tolist()
    except:
        pass

    raise ValueError("Could not parse file as a numeric matrix. Please check the format.")

# --- Streamlit UI ---

st.set_page_config(page_title="Graph Viz Pro", layout="wide")

st.title("Interactive Graph Visualizer")
st.markdown("Upload a matrix to visualize. Supports **Adjacency**, **Incidence**, and **Biadjacency**.")

# --- Sidebar ---
st.sidebar.header("1. Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
st.sidebar.markdown("**OR**")
matrix_input = st.sidebar.text_area("Paste Python List", value="[[0, 1, 0], [1, 0, 1], [0, 1, 0]]", height=100)

st.sidebar.header("2. Logic")
kind_option = st.sidebar.selectbox("Matrix Type", options=["auto", "adjacency", "incidence", "biadjacency"])

# --- Visual Settings ---
with st.sidebar.expander("3. Visualization Settings (Large Data)"):
    user_figsize = st.slider("Figure Size (Inches)", 5, 50, 12)
    user_node_size = st.slider("Node Size", 10, 1000, 300)
    user_font_size = st.slider("Font Size", 4, 24, 8)
    user_dpi = st.number_input("Download DPI", value=300)

if st.sidebar.button("Generate Graph", type="primary"):
    try:
        # --- Data Parsing ---
        matrix_data = None
        if uploaded_file:
            matrix_data = load_matrix_smart(uploaded_file)
            st.success(f"Loaded {len(matrix_data)}x{len(matrix_data[0])} matrix.")
        elif matrix_input:
            matrix_data = ast.literal_eval(matrix_input)
        
        if not matrix_data:
            st.error("No data found.")
            st.stop()

        # --- Graph Generation ---
        G, detected_kind = matrix_to_graph(matrix_data, kind=kind_option)
        num_nodes = G.number_of_nodes()

        # --- Dynamic Layout Heuristic ---
        # Adjust spring tension (k) based on density to prevent overlap
        k_val = 1 / np.sqrt(num_nodes) if num_nodes > 0 else 0.5
        
        col_viz, col_stats = st.columns([4, 1])

        with col_viz:
            with st.spinner("Calculating layout..."):
                fig, ax = plt.subplots(figsize=(user_figsize, user_figsize))
                
                # Layout
                if detected_kind == "biadjacency":
                    U_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
                    pos = nx.bipartite_layout(G, U_nodes)
                else:
                    pos = nx.spring_layout(G, seed=42, k=k_val, iterations=50)

                # Draw
                is_directed = isinstance(G, nx.DiGraph)
                nx.draw_networkx_nodes(G, pos, node_size=user_node_size, node_color='lightblue', ax=ax)
                nx.draw_networkx_edges(G, pos, arrows=is_directed, alpha=0.5, ax=ax)
                
                # Only draw labels if font size is reasonable
                if user_font_size > 0:
                    nx.draw_networkx_labels(G, pos, font_size=user_font_size, ax=ax)

                ax.set_title(f"{detected_kind.capitalize()} Graph", fontsize=16)
                ax.axis('off')
                st.pyplot(fig)

                # --- Download ---
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=user_dpi, bbox_inches='tight')
                img_buffer.seek(0)
                st.download_button("📥 Download Image", img_buffer, "graph.png", "image/png")
                plt.close(fig)

        with col_stats:
            st.info("Statistics")
            st.write(f"**Nodes:** {num_nodes}")
            st.write(f"**Edges:** {G.number_of_edges()}")
            st.write(f"**Density:** {nx.density(G):.4f}")
            if not is_directed and nx.is_connected(G):
                st.write(f"**Diameter:** {nx.diameter(G)}")
            
            with st.expander("Raw Data"):
                st.write(np.array(matrix_data))

    except Exception as e:
        st.error(f"Error: {e}")
