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
def get_cycles_with_weights(G, target_weight=None):
    """
    Finds cycles and optionally filters them by edge weight.
    Returns:
        - valid_cycles: List of cycles (each cycle is a list of nodes)
        - cycle_edges: List of edges involved in these cycles
    """
    # Detect cycles (simple_cycles for DiGraph, cycle_basis for Graph)
    if G.is_directed():
        # WARNING: simple_cycles is expensive on large dense graphs
        raw_cycles = list(nx.simple_cycles(G)) 
    else:
        raw_cycles = nx.cycle_basis(G)

    valid_cycles = []
    cycle_edges = set()

    for cycle in raw_cycles:
        # Create edge list for this cycle
        # e.g., [0, 1, 2] -> [(0,1), (1,2), (2,0)]
        c_edges = []
        has_target_weight = False
        
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            
            # Handle MultiGraphs or simple Graphs
            if G.has_edge(u, v):
                data = G.get_edge_data(u, v)
                w = data.get('weight', 1) # Default weight 1 if missing
                c_edges.append((u, v))
                
                # Check weight condition
                if target_weight is not None:
                    # geometric tolerance comparison for floats
                    if np.isclose(w, target_weight):
                        has_target_weight = True
            
        # Filter Logic
        if target_weight is None:
            # No filter, accept all
            valid_cycles.append(cycle)
            cycle_edges.update(c_edges)
        elif has_target_weight:
            # Only keep cycle if it contains the specific weight
            valid_cycles.append(cycle)
            cycle_edges.update(c_edges)

    return valid_cycles, list(cycle_edges)

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
    """Robustly loads a matrix from CSV/Excel, handling headers and indices."""
    uploaded_file.seek(0)
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    def read_func(file, **kwargs):
        if 'csv' in file_type:
            return pd.read_csv(file, **kwargs)
        else:
            return pd.read_excel(file, **kwargs)

    # Strategy 1: Raw matrix
    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, header=None)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all():
            return df_numeric.values.tolist()
    except:
        pass

    # Strategy 2: With Index
    try:
        uploaded_file.seek(0)
        df = read_func(uploaded_file, index_col=0)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.notna().all().all() and df_numeric.shape[0] > 0:
            return df_numeric.values.tolist()
    except:
        pass
    
    # Strategy 3: Header, No Index
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

try:
    G_original, detected_kind = matrix_to_graph(matrix_data, kind=kind_option)
except Exception as e:
    st.error(f"Graph Build Error: {e}")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Standard Visualization", "Frobenius Analysis", "Cycles & Weights"])

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
                
                # --- DOWNLOADS ---
                c1, c2 = st.columns(2)
                with c1:
                    # PNG Download
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=user_dpi, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("📥 Download Image (PNG)", buf, "graph.png", "image/png")
                    plt.close(fig)
                
                with c2:
                    # DOT Download
                    if HAS_PYDOT:
                        dot_buf = io.StringIO()
                        try:
                            write_dot(G_original, dot_buf)
                            st.download_button("📄 Download Graphviz (.gv)", dot_buf.getvalue(), "graph.gv", "text/plain")
                        except Exception as dot_err:
                            st.error(f"Export failed: {dot_err}")
                    else:
                        st.warning("Install 'pydot' to enable .gv export")

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
    1. **Matrix View:** Reorders the matrix to Block Triangular Form.
    2. **Graph View:** Colors nodes by cohort to show structure.
    """)

    if st.button("Analyze Cohorts", type="primary", key="btn_frob"):
        
        # 1. Ensure Directed Graph
        if detected_kind == "biadjacency" or not G_original.is_directed():
            st.warning("Converting to Directed Graph for component analysis.")
            G_ana = G_original.to_directed()
        else:
            G_ana = G_original.copy() # Copy to avoid mutating original with colors

        # 2. Compute Cohorts
        try:
            C = nx.condensation(G_ana) 
        except Exception as e:
            st.error(f"Error calculating cohorts: {e}")
            st.stop()
        
        node_to_cohort = {}
        for cohort_id, nodes_in_cohort in C.nodes(data="members"):
            for node in nodes_in_cohort:
                node_to_cohort[node] = cohort_id
        
        num_cohorts = len(C.nodes())
        st.success(f"Detected {num_cohorts} Cohorts (Strongly Connected Components)")

        # 3. Topological Sort for Matrix
        try:
            cohort_order = list(nx.topological_sort(C))
        except:
            cohort_order = list(C.nodes()) 

        # 4. Reorder Nodes
        ordered_nodes = []
        cohort_boundaries = [0]
        for cid in cohort_order:
            members = sorted(list(C.nodes[cid]['members'])) 
            ordered_nodes.extend(members)
            cohort_boundaries.append(len(ordered_nodes))

        # 5. Build Permuted Matrix
        node_to_idx = {n: i for i, n in enumerate(ordered_nodes)}
        N = len(ordered_nodes)
        P_matrix = np.zeros((N, N))
        for u, v, data in G_ana.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                P_matrix[i, j] = 1 

        # --- Visualization ---
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.subheader("1. Frobenius Matrix")
            fig_m, ax_m = plt.subplots(figsize=(8, 8))
            ax_m.imshow(P_matrix, cmap='Greys', interpolation='none')
            for b in cohort_boundaries[1:-1]:
                ax_m.axhline(b-0.5, color='red', linewidth=0.5, alpha=0.5)
                ax_m.axvline(b-0.5, color='red', linewidth=0.5, alpha=0.5)
            ax_m.axis('off')
            st.pyplot(fig_m)

        with col_f2:
            st.subheader("2. Cohort Graph")
            cmap = plt.get_cmap('tab20')
            
            # Layout
            pos_super = nx.spring_layout(C, seed=42, k=2.0) 
            pos_final = {}
            for cid in C.nodes():
                center = pos_super[cid]
                members = C.nodes[cid]['members']
                subG = G_ana.subgraph(members)
                sub_pos = nx.spring_layout(subG, center=center, scale=0.3)
                pos_final.update(sub_pos)

            # Assign Colors & Draw
            draw_nodes = list(G_ana.nodes())
            node_colors = []
            
            # Update G_ana attributes for DOT export
            for n in draw_nodes:
                cid = node_to_cohort.get(n, 0)
                rgba = cmap(cid % 20)
                node_colors.append(rgba)
                
                # Set Graphviz attributes
                hex_color = mcolors.to_hex(rgba)
                G_ana.nodes[n]['style'] = 'filled'
                G_ana.nodes[n]['fillcolor'] = hex_color
                G_ana.nodes[n]['color'] = 'black'
                G_ana.nodes[n]['fontcolor'] = 'black'

            fig_g, ax_g = plt.subplots(figsize=(8, 8))
            nx.draw_networkx_nodes(G_ana, pos_final, nodelist=draw_nodes, node_size=100, node_color=node_colors, ax=ax_g)
            nx.draw_networkx_edges(G_ana, pos_final, alpha=0.2, arrows=True, ax=ax_g)
            
            # Legend
            legend_elements = []
            top_cohorts = sorted(C.nodes(data="members"), key=lambda x: len(x[1]), reverse=True)[:10]
            for i, (cid, members) in enumerate(top_cohorts):
                c = cmap(cid % 20)
                legend_elements.append(patches.Patch(facecolor=c, label=f'Cohort {cid}'))
            ax_g.legend(handles=legend_elements, loc='upper right', fontsize=8)
            ax_g.axis('off')
            st.pyplot(fig_g)
            
            # --- DOWNLOADS ---
            c3, c4 = st.columns(2)
            with c3:
                buf_c = io.BytesIO()
                plt.savefig(buf_c, format='png', dpi=300, bbox_inches='tight')
                buf_c.seek(0)
                st.download_button("📥 Download Image (PNG)", buf_c, "cohort_graph.png", "image/png")
                plt.close(fig_g)
            
            with c4:
                if HAS_PYDOT:
                    dot_buf_c = io.StringIO()
                    try:
                        write_dot(G_ana, dot_buf_c)
                        st.download_button("📄 Download Graphviz (.gv)", dot_buf_c.getvalue(), "cohorts.gv", "text/plain")
                    except Exception as dot_err_c:
                        st.error(f"Export failed: {dot_err_c}")
                else:
                    st.warning("Install 'pydot' to enable .gv export")
# ==========================================
# TAB 3: Cycles & Weights
# ==========================================
with tab3:
    st.markdown("### Cycle Analysis & Weighted Filters")
    
    # 1. Controls
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        target_w = st.number_input("Target Edge Weight", value=1.0, step=0.1)
        
    with col_c2:
        view_mode = st.radio("View Mode", 
                             ["Highlight Edges (Full Graph)", 
                              "Filter Graph (Cycles Only)"])

    with col_c3:
        layout_option = st.selectbox("Layout Algorithm", 
                                     ["spring (auto)", "circular", "shell", "kamada_kawai", "bipartite", "planar"])

    st.divider() # Visual separation

    # --- Helper: Layout Calculation ---
    def get_layout(graph_obj, algo):
        if graph_obj.number_of_nodes() == 0:
            return {}
        try:
            if algo == "spring (auto)":
                return nx.spring_layout(graph_obj, seed=42)
            elif algo == "circular":
                return nx.circular_layout(graph_obj)
            elif algo == "shell":
                return nx.shell_layout(graph_obj)
            elif algo == "kamada_kawai":
                return nx.kamada_kawai_layout(graph_obj)
            elif algo == "planar":
                return nx.planar_layout(graph_obj)
            elif algo == "bipartite":
                U_nodes = {n for n, d in graph_obj.nodes(data=True) if d.get("bipartite") == 0}
                if not U_nodes: 
                    U_nodes, _ = nx.bipartite.sets(graph_obj)
                return nx.bipartite_layout(graph_obj, U_nodes)
        except Exception as e:
            # Fallback for planar/bipartite errors
            return nx.spring_layout(graph_obj, seed=42)
        return nx.spring_layout(graph_obj, seed=42)

    # --- RENDERING LOGIC ---
    # We remove the button so it auto-updates. 
    # If the graph is heavy, we could use a checkbox "Enable Live Preview".
    
    fig_c, ax_c = plt.subplots(figsize=(10, 10))
    should_render = False

    # --- LOGIC A: HIGHLIGHT EDGES (Full Graph) ---
    if view_mode == "Highlight Edges (Full Graph)":
        
        pos_c = get_layout(G_original, layout_option)

        # Draw Nodes
        nx.draw_networkx_nodes(G_original, pos_c, node_color='lightgrey', node_size=400, ax=ax_c)
        nx.draw_networkx_labels(G_original, pos_c, ax=ax_c)

        # Separate Edges
        edges_target = []
        edges_other = []
        
        for u, v, data in G_original.edges(data=True):
            w = data.get('weight', 1)
            # Use isclose for float comparison
            if np.isclose(w, target_w):
                edges_target.append((u, v))
            else:
                edges_other.append((u, v))

        # Draw Edges
        nx.draw_networkx_edges(G_original, pos_c, edgelist=edges_other, 
                               edge_color='black', alpha=0.3, ax=ax_c)
        nx.draw_networkx_edges(G_original, pos_c, edgelist=edges_target, 
                               edge_color='red', width=2.5, ax=ax_c)
        
        # Labels
        edge_labels = nx.get_edge_attributes(G_original, 'weight')
        nx.draw_networkx_edge_labels(G_original, pos_c, edge_labels=edge_labels, ax=ax_c)
        
        should_render = True

    # --- LOGIC B: FILTER (Cycles containing weight) ---
    else:
        # Get cycles that contain at least one edge of target_w
        valid_cycles, valid_edges = get_cycles_with_weights(G_original, target_w)
        
        if not valid_cycles:
            st.warning(f"No cycles found containing edges with weight {target_w}.")
            should_render = False # Do not render empty plot
        else:
            st.success(f"Found {len(valid_cycles)} cycles containing weight {target_w}.")
            
            # Build Subgraph
            G_sub = G_original.edge_subgraph(valid_edges).copy()
            
            # Layout
            pos_sub = get_layout(G_sub, layout_option)
            
            # Draw
            nx.draw_networkx_nodes(G_sub, pos_sub, node_color='#ffcc00', node_size=500, ax=ax_c)
            nx.draw_networkx_labels(G_sub, pos_sub, ax=ax_c)
            
            e_target = []
            e_other = []
            for u, v, data in G_sub.edges(data=True):
                w = data.get('weight', 1)
                if np.isclose(w, target_w):
                    e_target.append((u, v))
                else:
                    e_other.append((u, v))

            nx.draw_networkx_edges(G_sub, pos_sub, edgelist=e_other, edge_color='black', style='dashed', ax=ax_c)
            nx.draw_networkx_edges(G_sub, pos_sub, edgelist=e_target, edge_color='red', width=3, ax=ax_c)
            
            edge_labels_sub = {e: G_sub.edges[e]['weight'] for e in G_sub.edges}
            nx.draw_networkx_edge_labels(G_sub, pos_sub, edge_labels=edge_labels_sub, font_color='red', ax=ax_c)
            
            should_render = True

    # --- FINAL RENDER ---
    if should_render:
        ax_c.axis('off')
        st.pyplot(fig_c)
        plt.close(fig_c) # Good practice to close figure to free memory
