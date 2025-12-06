import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
import pandas as pd
import io

# --- Helper Functions ---

def matrix_to_graph(M, kind="auto"):
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
        if df_numeric.notna().all().all():
            return df_numeric.values.tolist()
    except:
        pass

    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, index_col=0)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all() and df_numeric.shape[0] > 0:
            return df_numeric.values.tolist()
    except:
        pass
    
    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, header=0)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all():
            return df_numeric.values.tolist()
    except:
        pass

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

# Global Data Loader
matrix_data = None
try:
    if uploaded_file:
        matrix_data = load_matrix_smart(uploaded_file)
        st.sidebar.success(f"Loaded {len(matrix_data)}x{len(matrix_data[0])} matrix.")
    elif matrix_input:
        matrix_data = ast.literal_eval(matrix_input)
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

if not matrix_data:
    st.info("Please upload a file or enter data to begin.")
    st.stop()

# Build Graph
try:
    G_original, detected_kind = matrix_to_graph(matrix_data, kind=kind_option)
except Exception as e:
    st.error(f"Graph Build Error: {e}")
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["Standard Visualization", "Frobenius Analysis (Cohorts)"])

# ==========================================
# TAB 1: Standard Visualization
# ==========================================
with tab1:
    with st.expander("Visualization Settings", expanded=False):
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        user_figsize = col_s1.slider("Fig Size", 5, 50, 12, key="t1_size")
        user_node_size = col_s2.slider("Node Size", 10, 1000, 300, key="t1_node")
        user_font_size = col_s3.slider("Font Size", 0, 24, 8, key="t1_font")
        user_dpi = col_s4.number_input("DPI", value=150, key="t1_dpi")

    if st.button("Render Standard Graph", type="primary", key="btn_std"):
        num_nodes = G_original.number_of_nodes()
        k_val = 1 / np.sqrt(num_nodes) if num_nodes > 0 else 0.5
        
        col_viz, col_stats = st.columns([4, 1])
        
        with col_viz:
            with st.spinner("Rendering..."):
                fig, ax = plt.subplots(figsize=(user_figsize, user_figsize))
                
                if detected_kind == "biadjacency":
                    U_nodes = [n for n, d in G_original.nodes(data=True) if d.get("bipartite") == 0]
                    pos = nx.bipartite_layout(G_original, U_nodes)
                else:
                    pos = nx.spring_layout(G_original, seed=42, k=k_val, iterations=50)

                is_directed = isinstance(G_original, nx.DiGraph)
                nx.draw_networkx_nodes(G_original, pos, node_size=user_node_size, node_color='lightblue', ax=ax)
                nx.draw_networkx_edges(G_original, pos, arrows=is_directed, alpha=0.5, ax=ax)
                if user_font_size > 0:
                    nx.draw_networkx_labels(G_original, pos, font_size=user_font_size, ax=ax)
                
                ax.axis('off')
                st.pyplot(fig)
                
                # Download
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=user_dpi, bbox_inches='tight')
                buf.seek(0)
                st.download_button("Download Image", buf, "graph.png", "image/png")
                plt.close(fig)

        with col_stats:
            st.write(f"**Nodes:** {num_nodes}")
            st.write(f"**Edges:** {G_original.number_of_edges()}")
            st.write(f"**Type:** {detected_kind}")

# ==========================================
# TAB 2: Frobenius Form & Cohorts
# ==========================================
with tab2:
    st.markdown("""
    **Frobenius Normal Form Analysis:** Identifies **Cohorts** (Strongly Connected Components).  
    1. **Matrix View:** Reorders the matrix to Block Triangular Form. Diagonal blocks are cohorts.  
    2. **Graph View:** Colors nodes by cohort to show structure.
    """)

    if st.button("Analyze Cohorts", type="primary", key="btn_frob"):
        
        # 1. Ensure Directed Graph for SCC analysis
        if detected_kind == "biadjacency" or not G_original.is_directed():
            st.warning("Converting to Directed Graph for component analysis.")
            G_ana = G_original.to_directed()
        else:
            G_ana = G_original

        # 2. Compute Condensed Graph (Cohorts)
        # The nodes of C are the SCCs of G_ana
        C = nx.condensations.condensation(G_ana)
        
        # Mapping: Node -> Cohort ID
        node_to_cohort = {}
        for cohort_id, nodes_in_cohort in C.nodes(data="members"):
            for node in nodes_in_cohort:
                node_to_cohort[node] = cohort_id
        
        num_cohorts = len(C.nodes())
        st.success(f"Detected {num_cohorts} Cohorts (Strongly Connected Components)")

        # 3. Create Topological Sorting of Cohorts (for Matrix ordering)
        # In a condensation graph (DAG), we can sort topologically.
        try:
            cohort_order = list(nx.topological_sort(C))
        except:
            cohort_order = list(C.nodes()) # Fallback if cycle (shouldn't happen in condensation)

        # 4. Reorder Nodes for Matrix
        ordered_nodes = []
        cohort_boundaries = [0]
        
        for cid in cohort_order:
            members = sorted(list(C.nodes[cid]['members'])) # Internal sort for neatness
            ordered_nodes.extend(members)
            cohort_boundaries.append(len(ordered_nodes))

        # 5. Build Permuted Matrix
        # Map original node labels to 0..N indices for matrix construction
        node_to_idx = {n: i for i, n in enumerate(ordered_nodes)}
        N = len(ordered_nodes)
        P_matrix = np.zeros((N, N))

        # Fill P_matrix based on edges in G_ana
        for u, v, data in G_ana.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                P_matrix[i, j] = 1 # Binary for visualization

        # --- Visualization Layout ---
        col_f1, col_f2 = st.columns(2)

        # PLOT A: Frobenius Matrix
        with col_f1:
            st.subheader("1. Frobenius Matrix Form")
            fig_m, ax_m = plt.subplots(figsize=(8, 8))
            ax_m.imshow(P_matrix, cmap='Greys', interpolation='none')
            
            # Draw lines for cohorts
            for b in cohort_boundaries[1:-1]:
                ax_m.axhline(b-0.5, color='red', linewidth=0.5, alpha=0.5)
                ax_m.axvline(b-0.5, color='red', linewidth=0.5, alpha=0.5)
            
            ax_m.set_title("Block Triangular Form\n(Red lines separate cohorts)", fontsize=10)
            ax_m.set_xlabel("Target Node (Reordered)")
            ax_m.set_ylabel("Source Node (Reordered)")
            st.pyplot(fig_m)

        # PLOT B: Cohort Graph
        with col_f2:
            st.subheader("2. Cohort Graph")
            
            # Assign colors to cohorts using a colormap
            cmap = plt.get_cmap('tab20')
            node_colors = []
            
            # We need to iterate nodes in the order G_ana stores them to match nx.draw
            draw_nodes = list(G_ana.nodes())
            for n in draw_nodes:
                cid = node_to_cohort.get(n, 0)
                # Hash cid to get a color index
                color = cmap(cid % 20)
                node_colors.append(color)

            fig_g, ax_g = plt.subplots(figsize=(8, 8))
            
            # Layout: Separate cohorts visually?
            # A good trick is to use the Condensation layout for the centers, then spring around them
            pos_super = nx.spring_layout(C, seed=42, k=2.0) # Layout of cohorts
            pos_final = {}
            
            # Position nodes around their cohort center
            for cid in C.nodes():
                center = pos_super[cid]
                members = C.nodes[cid]['members']
                # Create sub-graph for internal layout
                subG = G_ana.subgraph(members)
                # Small spring layout centered at 'center'
                sub_pos = nx.spring_layout(subG, center=center, scale=0.3)
                pos_final.update(sub_pos)

            nx.draw_networkx_nodes(G_ana, pos_final, node_size=100, node_color=node_colors, ax=ax_g)
            nx.draw_networkx_edges(G_ana, pos_final, alpha=0.2, arrows=True, ax=ax_g)
            
            # Legend for top 10 largest cohorts
            legend_elements = []
            top_cohorts = sorted(C.nodes(data="members"), key=lambda x: len(x[1]), reverse=True)[:10]
            for i, (cid, members) in enumerate(top_cohorts):
                c = cmap(cid % 20)
                legend_elements.append(patches.Patch(facecolor=c, label=f'Cohort {cid} ({len(members)} nodes)'))
            
            ax_g.legend(handles=legend_elements, loc='upper right', fontsize=8)
            ax_g.set_title("Network Colored by Cohort")
            ax_g.axis('off')
            st.pyplot(fig_g)
            
            # Download Logic for Cohort Graph
            buf_c = io.BytesIO()
            plt.savefig(buf_c, format='png', dpi=300, bbox_inches='tight')
            buf_c.seek(0)
            st.download_button("Download Cohort Graph", buf_c, "cohort_graph.png", "image/png")
            plt.close(fig_g)
