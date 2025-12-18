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

def get_cycles_general(G, target_weight=None, weight_attr='weight', source_node=None):
    """
    Finds cycles based on weight criteria, with optional node filtering.
    """
    def check_match(val, target):
        if target is None: return True 
        try:
            return np.isclose(float(val), float(target))
        except:
            return str(val) == str(target)

    # 1. Identify "Target Edges"
    target_edges_set = set()
    for u, v, data in G.edges(data=True):
        val = data.get(weight_attr, 1)
        
        is_connected = True
        if source_node is not None:
            if G.is_directed():
                if u != source_node: is_connected = False
            else:
                if u != source_node and v != source_node: is_connected = False
        
        if is_connected and check_match(val, target_weight):
            target_edges_set.add((u, v))
            if not G.is_directed():
                target_edges_set.add((v, u))

    if target_weight is not None and not target_edges_set:
        return [], [], []

    # 2. Cycle Detection
    if G.is_directed():
        cycle_gen = nx.simple_cycles(G)
    else:
        cycle_gen = nx.cycle_basis(G)

    valid_cycles = []
    cycle_edges_viz = set()

    # 3. Filter Cycles
    max_cycles = 5000 
    count = 0

    for cycle in cycle_gen:
        if source_node is not None and source_node not in cycle:
            continue

        has_target_edge = False
        c_edges = []
        
        for k in range(len(cycle)):
            u, v = cycle[k], cycle[(k + 1) % len(cycle)]
            c_edges.append((u, v))
            
            if target_weight is None or (u, v) in target_edges_set:
                has_target_edge = True
        
        if has_target_edge:
            valid_cycles.append(cycle)
            for e in c_edges:
                cycle_edges_viz.add(e)
                if not G.is_directed():
                    cycle_edges_viz.add((e[1], e[0]))
            
            count += 1
            if count >= max_cycles:
                break

    return valid_cycles, list(cycle_edges_viz), list(target_edges_set)

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

def process_uploaded_file(uploaded_file):
    """Separates headers (encoding) from numerical data."""
    uploaded_file.seek(0)
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if 'csv' in file_type:
            df = pd.read_csv(uploaded_file, index_col=0)
        else:
            df = pd.read_excel(uploaded_file, index_col=0)
        
        row_labels = list(df.index)
        col_labels = list(df.columns)
        
        encoding_data = []
        for i, label in enumerate(row_labels):
            encoding_data.append({'Node_ID': i, 'Original_Label': label, 'Type': 'Row'})
        
        if row_labels != col_labels:
            for j, label in enumerate(col_labels):
                 encoding_data.append({'Node_ID': j, 'Original_Label': label, 'Type': 'Column'})
        
        encoding_df = pd.DataFrame(encoding_data)

        df_clean = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        if df_clean.shape[0] == 0: return None, None, "File appears empty."

        return df_clean.values.tolist(), encoding_df, None

    except Exception as e:
        try:
            uploaded_file.seek(0)
            if 'csv' in file_type:
                df = pd.read_csv(uploaded_file, header=None)
            else:
                df = pd.read_excel(uploaded_file, header=None)
            
            df_clean = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            encoding_data = [{'Node_ID': i, 'Original_Label': f"Node_{i}", 'Type': 'Index'} for i in range(len(df_clean))]
            encoding_df = pd.DataFrame(encoding_data)
            
            return df_clean.values.tolist(), encoding_df, None
        except Exception as e2:
            return None, None, str(e2)

# --- UI Setup ---

st.set_page_config(page_title="Graph Viz Pro", layout="wide")
st.title("Interactive Graph Visualizer")

# --- Sidebar ---
st.sidebar.header("1. Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel (Stripped Headers)", type=["csv", "xlsx", "xls"])
st.sidebar.markdown("**OR**")
matrix_input = st.sidebar.text_area("Paste Python List", value="[[0, 1, 0], [1, 0, 1], [0, 1, 0]]", height=100)

st.sidebar.header("2. Logic")
kind_option = st.sidebar.selectbox("Matrix Type", options=["auto", "adjacency", "incidence", "biadjacency"])

# --- Main Logic ---

matrix_data = None
encoding_df = None

try:
    if uploaded_file:
        matrix_data, encoding_df, error_msg = process_uploaded_file(uploaded_file)
        if error_msg:
            st.error(f"Error processing file: {error_msg}")
            st.stop()
        st.sidebar.success(f"Loaded {len(matrix_data)}x{len(matrix_data[0])} matrix.")
        if encoding_df is not None:
            csv_enc = encoding_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button("📥 Download Encoding Key", csv_enc, "node_encoding.csv", "text/csv")
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
        show_weights_t1 = st.checkbox("Show Edge Weights", value=True, key="t1_weights")

    if st.button("Render Standard Graph", type="primary", key="btn_std"):
        num_nodes = G_original.number_of_nodes()
        k_val = 1 / np.sqrt(num_nodes) if num_nodes > 0 else 0.5
        
        col_viz, col_stats = st.columns([4, 1])
        
        with col_viz:
            with st.spinner("Rendering..."):
                fig, ax = plt.subplots(figsize=(user_figsize, user_figsize))
                
                # Layout
                if detected_kind == "biadjacency":
                    U_nodes = [n for n, d in G_original.nodes(data=True) if d.get("bipartite") == 0]
                    pos = nx.bipartite_layout(G_original, U_nodes)
                else:
                    pos = nx.spring_layout(G_original, seed=42, k=k_val, iterations=50)

                is_directed = isinstance(G_original, nx.DiGraph)
                
                # FIX: connectionstyle must be a valid string even if undirected
                conn_style = "arc3,rad=0.1" if is_directed else "arc3,rad=0.0"
                
                # Draw Nodes
                nx.draw_networkx_nodes(G_original, pos, node_size=user_node_size, node_color='lightblue', ax=ax)
                
                # Draw Edges (with curve if directed)
                nx.draw_networkx_edges(G_original, pos, arrows=is_directed, alpha=0.5, 
                                       connectionstyle=conn_style, ax=ax)
                
                # Draw Node Labels
                if user_font_size > 0:
                    nx.draw_networkx_labels(G_original, pos, font_size=user_font_size, ax=ax)
                
                # Draw Edge Weights (Small Font)
                if show_weights_t1:
                    edge_labels = nx.get_edge_attributes(G_original, 'weight')
                    fmt_labels = {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in edge_labels.items()}
                    nx.draw_networkx_edge_labels(G_original, pos, edge_labels=fmt_labels, 
                                                 font_size=8, label_pos=0.5, ax=ax)
                
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                
                # Downloads
                c1, c2 = st.columns(2)
                with c1:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=user_dpi, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("📥 Download Image (PNG)", buf, "graph.png", "image/png")
                with c2:
                    if HAS_PYDOT:
                        dot_buf = io.StringIO()
                        try:
                            write_dot(G_original, dot_buf)
                            st.download_button("📄 Download Graphviz (.gv)", dot_buf.getvalue(), "graph.gv", "text/plain")
                        except Exception: pass

        with col_stats:
            st.write(f"**Nodes:** {num_nodes}")
            st.write(f"**Edges:** {G_original.number_of_edges()}")

# ==========================================
# TAB 2: Frobenius Form & Cohorts
# ==========================================
with tab2:
    st.markdown("""
    **Frobenius Normal Form Analysis:** Identifies **Cohorts** (Strongly Connected Components).  
    """)
    
    col_t2_1, col_t2_2 = st.columns([1, 2])
    with col_t2_1:
        frob_mode = st.selectbox(
            "Matrix Interpretation for Analysis", 
            ["Same as Sidebar", "adjacency", "incidence", "biadjacency"],
            key="frob_mode"
        )

    if st.button("Analyze Cohorts", type="primary", key="btn_frob"):
        
        # --- NEW LOGIC: Handle Biadjacency -> Adjacency Conversion ---
        current_target_kind = frob_mode if frob_mode != "Same as Sidebar" else detected_kind

        if current_target_kind == "biadjacency":
            st.info("🔄 Treating Biadjacency Matrix as Adjacency for Cohort Analysis (creating equivalent graph).")
            
            # Check dimensions to decide strategy
            M = np.array(matrix_data)
            rows, cols = M.shape
            
            if rows == cols:
                # Square Matrix: Treat directly as Adjacency (Node i -> Node j)
                try:
                    G_ana, _ = matrix_to_graph(matrix_data, kind="adjacency")
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
                    st.stop()
            else:
                # Rectangular Matrix: Create Bipartite Adjacency Matrix
                # Block Matrix: [[0, M], [0, 0]] (Directed U->V) or [[0, M], [M.T, 0]] (Undirected)
                # Frobenius usually requires square, so we build the square bipartite adjacency.
                st.warning(f"Matrix is rectangular ({rows}x{cols}). Building Bipartite Adjacency Graph.")
                
                # Construct Bipartite Graph explicitly
                G_ana = nx.DiGraph()
                # Row nodes 0..rows-1, Col nodes rows..rows+cols-1
                G_ana.add_nodes_from(range(rows + cols))
                
                for r in range(rows):
                    for c in range(cols):
                        val = M[r][c]
                        if val != 0:
                            # Edge from Row Node r -> Col Node (rows + c)
                            G_ana.add_edge(r, rows + c, weight=val)

        elif frob_mode != "Same as Sidebar":
            # Manual override (e.g., forcing incidence)
            try:
                G_ana, _ = matrix_to_graph(matrix_data, kind=frob_mode)
            except Exception as e:
                st.error(f"Error rebuilding graph as {frob_mode}: {e}")
                st.stop()
        else:
            # Default behavior (use globally detected graph)
            if detected_kind == "biadjacency" or not G_original.is_directed():
                 G_ana = G_original.to_directed()
            else:
                 G_ana = G_original.copy()

        # --- Proceed with Frobenius Analysis ---
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
        st.success(f"Detected {num_cohorts} Cohorts")

        try:
            cohort_order = list(nx.topological_sort(C))
        except:
            cohort_order = list(C.nodes()) 

        ordered_nodes = []
        cohort_boundaries = [0]
        for cid in cohort_order:
            members = sorted(list(C.nodes[cid]['members'])) 
            ordered_nodes.extend(members)
            cohort_boundaries.append(len(ordered_nodes))

        node_to_idx = {n: i for i, n in enumerate(ordered_nodes)}
        N = len(ordered_nodes)
        P_matrix = np.zeros((N, N))
        for u, v, data in G_ana.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                P_matrix[i, j] = 1 

        col_f1, col_f2 = st.columns(2)

        # --- 1. Frobenius Matrix ---
        with col_f1:
            st.subheader("1. Frobenius Matrix")
            fig_m, ax_m = plt.subplots(figsize=(8, 8))
            ax_m.imshow(P_matrix, cmap='Blues', interpolation='none', aspect='equal')
            
            for b in cohort_boundaries[1:-1]:
                ax_m.axhline(b-0.5, color='red', linewidth=1.0)
                ax_m.axvline(b-0.5, color='red', linewidth=1.0)
            
            ax_m.set_xticks(range(N))
            ax_m.set_yticks(range(N))
            ax_m.set_xticklabels(ordered_nodes, rotation=90, fontsize=6)
            ax_m.set_yticklabels(ordered_nodes, fontsize=6)
            ax_m.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            
            st.pyplot(fig_m)
            plt.close(fig_m)

        # --- 2. Cohort Graph ---
        with col_f2:
            st.subheader("2. Cohort Graph")
            cmap = plt.get_cmap('tab20')
            pos_super = nx.spring_layout(C, seed=42, k=2.0) 
            pos_final = {}
            for cid in C.nodes():
                center = pos_super[cid]
                members = C.nodes[cid]['members']
                subG = G_ana.subgraph(members)
                sub_pos = nx.spring_layout(subG, center=center, scale=0.3)
                pos_final.update(sub_pos)

            draw_nodes = list(G_ana.nodes())
            node_colors = [cmap(node_to_cohort.get(n, 0) % 20) for n in draw_nodes]

            fig_g, ax_g = plt.subplots(figsize=(8, 8))
            nx.draw_networkx_nodes(G_ana, pos_final, nodelist=draw_nodes, node_size=100, node_color=node_colors, ax=ax_g)
            
            conn_style = "arc3,rad=0.1"
            nx.draw_networkx_edges(G_ana, pos_final, alpha=0.3, arrows=True, connectionstyle=conn_style, ax=ax_g)
            
            nx.draw_networkx_labels(G_ana, pos_final, font_size=8, ax=ax_g)
            
            legend_elements = []
            top_cohorts = sorted(C.nodes(data="members"), key=lambda x: len(x[1]), reverse=True)[:10]
            for i, (cid, members) in enumerate(top_cohorts):
                c = cmap(cid % 20)
                legend_elements.append(patches.Patch(facecolor=c, label=f'Cohort {cid}'))
            ax_g.legend(handles=legend_elements, loc='upper right', fontsize=8)
            ax_g.axis('off')
            st.pyplot(fig_g)
            plt.close(fig_g)

# ==========================================
# TAB 3: Cycles & Weights
# ==========================================
with tab3:
    st.markdown("### 🔍 Specific Weight & Cycle Analysis")
    
    all_edge_keys = set()
    for u, v, data in G_original.edges(data=True):
        all_edge_keys.update(data.keys())
    
    available_cols = list(all_edge_keys)
    default_ix = available_cols.index('weight') if 'weight' in available_cols else (0 if available_cols else None)

    # --- 1. Selection Controls ---
    col_ctrl, col_viz = st.columns([1, 3])

    with col_ctrl:
        st.subheader("1. Criteria")
        
        if available_cols:
            weight_col = st.selectbox("Attribute / Column", available_cols, index=default_ix)
        else:
            weight_col = None

        use_specific_node = st.checkbox("Filter: Must pass through Node", value=True)
        all_nodes = sorted(list(G_original.nodes()), key=lambda x: (isinstance(x, str), x))
        
        source_u = None
        if use_specific_node:
            source_u = st.selectbox("Select Through Node", all_nodes)

        out_values = set()
        for u, v, data in G_original.edges(data=True):
            val = data.get(weight_col, 1) if weight_col else 1
            should_add = False
            if use_specific_node:
                if G_original.is_directed():
                    if u == source_u: should_add = True
                else:
                    if u == source_u or v == source_u: should_add = True
            else:
                should_add = True
            if should_add:
                out_values.add(val)

        try:
            sorted_vals = sorted(list(out_values), key=lambda x: float(x))
        except:
            sorted_vals = sorted(list(out_values), key=lambda x: str(x))
        
        if not sorted_vals:
            st.warning("No data for selection.")
            sel_val = None
        else:
            sel_val = st.selectbox(f"Select Value ({weight_col})", sorted_vals)
        
        st.divider()
        st.subheader("2. Appearance")
        show_labels = st.checkbox("Show Node Labels", value=True)
        show_weights_t3 = st.checkbox("Show Edge Weights", value=True, key="t3_weights")
        label_size = st.slider("Label Size", 5, 20, 10, disabled=not show_labels)
        
        layout_algo = st.selectbox("Layout", ["spring", "circular", "kamada_kawai", "shell", "bipartite", "planar"])
        
        view_mode = st.radio("Mode", ["Highlight (Full)", "Filter (Cycles Only)"])

    # --- Helper: Layout ---
    def get_layout(graph_obj, algo):
        if graph_obj.number_of_nodes() == 0: return {}
        try:
            sorted_nodes = sorted(list(graph_obj.nodes()), key=lambda x: (isinstance(x, str), x))
        except:
            sorted_nodes = list(graph_obj.nodes())

        try:
            if algo == "spring": return nx.spring_layout(graph_obj, seed=42, k=0.5)
            elif algo == "circular": 
                pos = {}
                if len(sorted_nodes) > 0:
                    angle_step = 2 * np.pi / len(sorted_nodes)
                    for i, node in enumerate(sorted_nodes):
                        theta = np.pi/2 - i * angle_step
                        pos[node] = np.array([np.cos(theta), np.sin(theta)])
                return pos
            elif algo == "kamada_kawai": return nx.kamada_kawai_layout(graph_obj)
            elif algo == "shell": return nx.shell_layout(graph_obj, nlist=[sorted_nodes])
            elif algo == "planar": return nx.planar_layout(graph_obj)
            elif algo == "bipartite":
                U_nodes = {n for n, d in graph_obj.nodes(data=True) if d.get("bipartite") == 0}
                if not U_nodes: U_nodes, _ = nx.bipartite.sets(graph_obj)
                return nx.bipartite_layout(graph_obj, sorted(list(U_nodes), key=lambda x: (isinstance(x, str), x)))
        except:
            return nx.spring_layout(graph_obj, seed=42)
        return nx.spring_layout(graph_obj, seed=42)

    def is_val_match(d_val, t_val):
        try:
            return np.isclose(float(d_val), float(t_val))
        except:
            return str(d_val) == str(t_val)

    # --- 2. Visualization Area ---
    with col_viz:
        if sel_val is None:
            st.info("No data.")
        else:
            try:
                fig_c, ax_c = plt.subplots(figsize=(8, 8))
                use_arrows = bool(G_original.is_directed())
                conn_style = "arc3,rad=0.1" if use_arrows else "arc3,rad=0.0"

                # === Logic A: Highlight (Full Graph) ===
                if view_mode == "Highlight (Full)":
                    pos = get_layout(G_original, layout_algo)
                    nx.draw_networkx_nodes(G_original, pos, node_color='#E0E0E0', node_size=500, ax=ax_c)
                    
                    e_target = []
                    e_other = []
                    
                    for u, v, data in G_original.edges(data=True):
                        w = data.get(weight_col, 1)
                        is_match = is_val_match(w, sel_val)
                        
                        if use_specific_node:
                            is_connected = (u == source_u) if G_original.is_directed() else (u == source_u or v == source_u)
                            if is_connected and is_match: e_target.append((u, v))
                            else: e_other.append((u, v))
                        else:
                            if is_match: e_target.append((u, v))
                            else: e_other.append((u, v))

                    if e_other:
                        nx.draw_networkx_edges(G_original, pos, edgelist=e_other, edge_color='#B0B0B0', 
                                               alpha=0.4, arrows=use_arrows, connectionstyle=conn_style, ax=ax_c)
                    if e_target:
                        nx.draw_networkx_edges(G_original, pos, edgelist=e_target, edge_color='#FF4B4B', 
                                               width=2.5, arrows=use_arrows, connectionstyle=conn_style, ax=ax_c)
                    
                    if show_labels:
                        nx.draw_networkx_labels(G_original, pos, font_size=label_size, ax=ax_c)
                    if show_weights_t3 and weight_col:
                         edge_lbls = {e: G_original.edges[e].get(weight_col, '') for e in G_original.edges}
                         nx.draw_networkx_edge_labels(G_original, pos, edge_labels=edge_lbls, font_size=8, label_pos=0.5, ax=ax_c)

                # === Logic B: Filter Cycles (With Length Filter) ===
                else:
                    valid_cycles, valid_edges, target_edges = get_cycles_general(
                        G_original, target_weight=sel_val, weight_attr=weight_col, 
                        source_node=source_u if use_specific_node else None
                    )
                    
                    if not valid_cycles:
                        st.warning("No cycles found matching criteria.")
                        ax_c.axis('off')
                    else:
                        cycle_lengths = [len(c) for c in valid_cycles]
                        unique_lengths = sorted(list(set(cycle_lengths)))
                        
                        selected_lengths = st.multiselect(
                            "Filter by Cycle Length:", 
                            options=unique_lengths, 
                            default=unique_lengths
                        )
                        
                        filtered_cycles = [c for c in valid_cycles if len(c) in selected_lengths]
                        
                        if not filtered_cycles:
                            st.info("No cycles of selected length.")
                            ax_c.axis('off')
                        else:
                            cmap = plt.get_cmap('tab10')
                            len_to_color = {l: cmap(i % 10) for i, l in enumerate(unique_lengths)}
                            
                            filtered_edges = set()
                            for c in filtered_cycles:
                                for k in range(len(c)):
                                    filtered_edges.add((c[k], c[(k+1)%len(c)]))
                                    if not G_original.is_directed():
                                        filtered_edges.add((c[(k+1)%len(c)], c[k]))
                            
                            G_sub = G_original.edge_subgraph(list(filtered_edges)).copy()
                            pos = get_layout(G_sub, layout_algo)
                            
                            nx.draw_networkx_nodes(G_sub, pos, node_color='lightgrey', node_size=600, edgecolors='black', ax=ax_c)
                            
                            for length in selected_lengths:
                                color = len_to_color[length]
                                cycles_of_len = [c for c in filtered_cycles if len(c) == length]
                                
                                for cycle in cycles_of_len:
                                    c_edges = []
                                    for k in range(len(cycle)):
                                        c_edges.append((cycle[k], cycle[(k + 1) % len(cycle)]))
                                    
                                    nx.draw_networkx_edges(G_sub, pos, edgelist=c_edges, edge_color=[color], 
                                                           width=2.5, arrows=use_arrows, connectionstyle=conn_style, ax=ax_c)

                            if show_labels:
                                nx.draw_networkx_labels(G_sub, pos, font_size=label_size, font_weight='bold', ax=ax_c)
                            
                            if show_weights_t3 and weight_col:
                                edge_lbls = {e: G_sub.edges[e].get(weight_col, '') for e in G_sub.edges}
                                nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_lbls, font_color='black', font_size=8, ax=ax_c)
                            
                            legend_handles = [patches.Patch(color=len_to_color[l], label=f'Length {l}') for l in selected_lengths]
                            ax_c.legend(handles=legend_handles, loc='upper right')

                            st.divider()
                            st.subheader("📋 Cycle Data Lookup")
                            lookup_data = []
                            for i, cycle in enumerate(filtered_cycles):
                                length = len(cycle)
                                entries = []
                                for k in range(len(cycle)):
                                    u_n, v_n = cycle[k], cycle[(k + 1) % len(cycle)]
                                    data = G_original.get_edge_data(u_n, v_n)
                                    val = data.get(weight_col, 'N/A')
                                    entries.append(f"({u_n}→{v_n}: {val})")
                                
                                color_hex = mcolors.to_hex(len_to_color[length])
                                lookup_data.append({
                                    "Cycle ID": i+1,
                                    "Length": length,
                                    "Path": " → ".join(map(str, cycle + [cycle[0]])),
                                    "Entries": ", ".join(entries),
                                    "Color": color_hex
                                })
                            
                            df_lookup = pd.DataFrame(lookup_data)
                            def highlight_color(row):
                                return [f'background-color: {row["Color"]}; color: white' if col == 'Length' else '' for col in row.index]
                            st.dataframe(df_lookup.style.apply(highlight_color, axis=1))

                ax_c.axis('off')
                st.pyplot(fig_c)
                plt.close(fig_c)

            except Exception as viz_err:
                st.error(f"Visualization Error: {viz_err}")
                plt.close(fig_c)
