import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast
import pandas as pd
import io  # Required for handling image buffers

# --- Core Logic (Unchanged) ---

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

# --- Visual Settings (New) ---
with st.sidebar.expander("3. Visualization Settings (Large Data)"):
    # Default calculated later, but user can override
    st.markdown("Adjust these if the graph is too crowded.")
    user_figsize = st.slider("Figure Size (Inches)", min_value=5, max_value=50, value=10)
    user_node_size = st.slider("Node Size", min_value=10, max_value=2000, value=500)
    user_font_size = st.slider("Font Size", min_value=4, max_value=24, value=10)
    user_dpi = st.number_input("Download DPI (Resolution)", value=300, min_value=72, max_value=600)

if st.sidebar.button("Generate Graph", type="primary"):
    try:
        # --- Data Parsing ---
        matrix_data = None
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=None)
            else:
                df = pd.read_excel(uploaded_file, header=None)
            matrix_data = df.to_numpy().tolist()
            st.success(f"Loaded {len(matrix_data)}x{len(matrix_data[0])} matrix.")
        elif matrix_input:
            matrix_data = ast.literal_eval(matrix_input)
        
        if not matrix_data:
            st.error("No data found.")
            st.stop()

        # --- Graph Generation ---
        G, detected_kind = matrix_to_graph(matrix_data, kind=kind_option)
        num_nodes = G.number_of_nodes()

        # --- Dynamic Sizing Heuristics ---
        # If user didn't manually change the slider from default (10), we try to auto-scale
        # But since we have manual sliders, we trust the slider values.
        # Below logic sets layout spacing strength based on density
        k_val = 1 / np.sqrt(num_nodes) if num_nodes > 0 else None  # Optimal distance for spring layout

        col_viz, col_stats = st.columns([4, 1])

        with col_viz:
            with st.spinner("Calculating layout..."):
                # Create Figure
                fig, ax = plt.subplots(figsize=(user_figsize, user_figsize))
                
                # Layout Algorithms
                if detected_kind == "biadjacency":
                    U_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
                    pos = nx.bipartite_layout(G, U_nodes)
                else:
                    # k controls spacing; iterations helps large graphs settle
                    pos = nx.spring_layout(G, seed=42, k=k_val, iterations=50)

                # Draw
                is_directed = isinstance(G, nx.DiGraph)
                nx.draw_networkx_nodes(G, pos, node_size=user_node_size, node_color='lightblue', ax=ax)
                nx.draw_networkx_edges(G, pos, arrows=is_directed, alpha=0.6, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=user_font_size, ax=ax)

                # Edge Labels (Only if graph is small enough, otherwise it's messy)
                if num_nodes < 30:
                    edge_labels = nx.get_edge_attributes(G, "weight")
                    if edge_labels:
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=user_font_size-2, ax=ax)

                ax.set_title(f"{detected_kind.capitalize()} Graph Representation", fontsize=16)
                ax.axis('off')

                # Render to Streamlit
                st.pyplot(fig)

                # --- Download Logic ---
                # Save figure to in-memory buffer
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=user_dpi, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download High-Res Image",
                    data=img_buffer,
                    file_name="graph_visualization.png",
                    mime="image/png"
                )
                
                # Clean up memory
                plt.close(fig)

        with col_stats:
            st.info("Statistics")
            st.write(f"**Nodes:** {num_nodes}")
            st.write(f"**Edges:** {G.number_of_edges()}")
            st.write(f"**Density:** {nx.density(G):.4f}")
            if nx.is_connected(G.to_undirected()):
                st.write(f"**Diameter:** {nx.diameter(G.to_undirected())}")
            else:
                st.write("**Diameter:** Inf (Disconnected)")

    except Exception as e:
        st.error(f"Error: {e}")
