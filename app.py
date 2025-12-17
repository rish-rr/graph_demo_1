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
def get_cycles_with_node_weight(G, source_node, target_weight):
    """
    Finds cycles containing any edge starting at 'source_node' with weight 'target_weight'.
    """
    if G.is_directed():
        raw_cycles = list(nx.simple_cycles(G))
    else:
        raw_cycles = nx.cycle_basis(G)

    valid_cycles = []
    cycle_edges = set()
    
    # Identify target edges: (source_node, v) where weight == target_weight
    target_edges = []
    for u, v, data in G.edges(data=True):
        # Direction matters: u must be the source_node
        if u == source_node: 
            w = data.get('weight', 1)
            # Check weight match
            is_match = False
            try:
                if np.isclose(float(w), float(target_weight)): is_match = True
            except:
                if w == target_weight: is_match = True
            
            if is_match:
                target_edges.append((u, v))

    # Optimization: Set of target edges for fast lookup
    target_edge_set = set(target_edges)
    if not G.is_directed():
        # For undirected, (u, v) is same as (v, u). Add reverse to set.
        for u, v in target_edges:
            target_edge_set.add((v, u))

    for cycle in raw_cycles:
        c_edges = []
        found_target = False
        
        for k in range(len(cycle)):
            n1 = cycle[k]
            n2 = cycle[(k + 1) % len(cycle)]
            c_edges.append((n1, n2))
            
            # Check if this edge is one of our interesting edges
            if (n1, n2) in target_edge_set:
                found_target = True
        
        if found_target:
            valid_cycles.append(cycle)
            cycle_edges.update(c_edges)

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
# TAB 3: Cycles & Node-Weight Filter
# ==========================================
with tab3:
    st.markdown("### 🔍 Specific Weight Analysis")
    
    # --- 1. Selection Controls ---
    col_ctrl, col_viz = st.columns([1, 3])

    with col_ctrl:
        st.subheader("1. Criteria")
        all_nodes = list(G_original.nodes())
        source_u = st.selectbox("Select Source Node", all_nodes)

        # Find weights outgoing from this node
        out_weights = set()
        for u, v, data in G_original.edges(data=True):
            # For undirected, check both directions if needed, but usually we treat source_u as the starting point
            if G_original.is_directed():
                if u == source_u:
                    out_weights.add(data.get('weight', 1))
            else:
                if u == source_u or v == source_u:
                    out_weights.add(data.get('weight', 1))

        # Sort weights for display
        sorted_weights = sorted(list(out_weights), key=lambda x: float(x) if isinstance(x, (int, float)) else str(x))
        
        if not sorted_weights:
            st.warning("Node has no connecting edges.")
            sel_weight = None
        else:
            sel_weight = st.selectbox("Select Weight", sorted_weights)
        
        st.divider()
        st.subheader("2. Appearance")
        show_labels = st.checkbox("Show Labels", value=True)
        label_size = st.slider("Label Size", 5, 20, 10, disabled=not show_labels)
        
        layout_algo = st.selectbox("Layout", ["spring", "circular", "kamada_kawai", "shell"])
        view_mode = st.radio("Mode", ["Highlight (Full)", "Filter (Cycles)"])

    # --- Helper: Layout ---
    def get_layout(graph_obj, algo):
        if graph_obj.number_of_nodes() == 0: return {}
        try:
            if algo == "spring": return nx.spring_layout(graph_obj, seed=42, k=0.5)
            elif algo == "circular": return nx.circular_layout(graph_obj)
            elif algo == "kamada_kawai": return nx.kamada_kawai_layout(graph_obj)
            elif algo == "shell": return nx.shell_layout(graph_obj)
        except:
            return nx.spring_layout(graph_obj, seed=42)
        return nx.spring_layout(graph_obj, seed=42)

    # --- 2. Visualization Area ---
    with col_viz:
        if sel_weight is None:
            st.info("Select a node with edges to begin.")
        else:
            fig_c, ax_c = plt.subplots(figsize=(8, 8))
            
            # --- Logic A: Full Graph Highlight ---
            if view_mode == "Highlight (Full)":
                pos = get_layout(G_original, layout_algo)
                
                # Draw Nodes
                nx.draw_networkx_nodes(G_original, pos, node_color='#E0E0E0', node_size=500, ax=ax_c)
                
                # Edges: Split into "Target" (connected to source with weight) and "Others"
                e_target = []
                e_other = []
                
                for u, v, data in G_original.edges(data=True):
                    w = data.get('weight', 1)
                    
                    # Check connection matches selection
                    is_match = False
                    is_connected_to_source = False
                    
                    if G_original.is_directed():
                        if u == source_u: is_connected_to_source = True
                    else:
                        if u == source_u or v == source_u: is_connected_to_source = True

                    if is_connected_to_source:
                        try:
                            if np.isclose(float(w), float(sel_weight)): is_match = True
                        except:
                            if w == sel_weight: is_match = True
                    
                    if is_match:
                        e_target.append((u, v))
                    else:
                        e_other.append((u, v))

                # Style: Curved edges if directed
                conn_style = "arc3,rad=0.1" if G_original.is_directed() else None

                # Safely draw edges only if lists are not empty
                if e_other:
                    nx.draw_networkx_edges(G_original, pos, edgelist=e_other, edge_color='#B0B0B0', 
                                           alpha=0.4, connectionstyle=conn_style, ax=ax_c)
                if e_target:
                    nx.draw_networkx_edges(G_original, pos, edgelist=e_target, edge_color='#FF4B4B', 
                                           width=2.5, connectionstyle=conn_style, ax=ax_c)
                
                if show_labels:
                    nx.draw_networkx_labels(G_original, pos, font_size=label_size, ax=ax_c)

            # --- Logic B: Filter Cycles ---
            else:
                valid_cycles, valid_edges, target_edges = get_cycles_with_node_weight(G_original, source_u, sel_weight)
                
                if not valid_cycles:
                    st.warning(f"No cycles found passing through {source_u} with weight {sel_weight}.")
                    ax_c.text(0.5, 0.5, "No Cycles Found", ha='center', va='center', transform=ax_c.transAxes)
                    ax_c.axis('off')
                else:
                    st.success(f"Found {len(valid_cycles)} cycles.")
                    
                    G_sub = G_original.edge_subgraph(valid_edges).copy()
                    pos = get_layout(G_sub, layout_algo)
                    
                    nx.draw_networkx_nodes(G_sub, pos, node_color='#FFD700', node_size=600, edgecolors='black', ax=ax_c)
                    
                    # Edges: Highlight the specific edges that triggered the filter
                    e_trigger = []
                    e_normal = []
                    
                    target_set = set(target_edges)
                    # For undirected, ensure both directions are in the set
                    if not G_original.is_directed():
                        for u, v in list(target_set):
                            target_set.add((v, u))
                    
                    for u, v in G_sub.edges():
                        if (u, v) in target_set:
                            e_trigger.append((u, v))
                        else:
                            e_normal.append((u, v))

                    conn_style = "arc3,rad=0.1" if G_original.is_directed() else None
                    
                    # FIX: Only call draw if the list is not empty
                    if e_normal:
                        nx.draw_networkx_edges(G_sub, pos, edgelist=e_normal, edge_color='black', 
                                               style='dashed', width=1, connectionstyle=conn_style, ax=ax_c)
                    
                    if e_trigger:
                        nx.draw_networkx_edges(G_sub, pos, edgelist=e_trigger, edge_color='#FF4B4B', 
                                               width=3, connectionstyle=conn_style, ax=ax_c)

                    if show_labels:
                        nx.draw_networkx_labels(G_sub, pos, font_size=label_size, font_weight='bold', ax=ax_c)

            ax_c.axis('off')
            st.pyplot(fig_c)
            plt.close(fig_c)
