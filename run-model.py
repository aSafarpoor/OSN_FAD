import numpy as np
import csv
import ast
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


###################################################################
# 1) Minimal Index Construction
###################################################################
def build_node_index(node_info_path, embed_path):
    """
    Build a minimal in-memory index:
      - node_id -> node_type
      - node_id -> file line (or offset) for embeddings
    """
    node_type_map = {}
    embedding_offset_map = {}

    # 1) Read node_information.csv to get node types
    with open(node_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading node info"):
            nid = int(row['node_id'])
            attr_dict = ast.literal_eval(row['attributes'])
            ntype = attr_dict.get('node_type', 'unknown')
            node_type_map[nid] = ntype

    # 2) Build line-offset map from embed_path (node_embeddings.csv)
    line_num = 0
    with open(embed_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading embedding offsets"):
            nid = int(row['node_id'])
            embedding_offset_map[nid] = line_num
            line_num += 1

    return node_type_map, embedding_offset_map


###################################################################
# 2) Node Labels
###################################################################
def load_labels(node_labels_path):
    """
    Reads node_id -> label from node_labels.csv.
    Returns labels_map: dict {node_id: label_str}
    """
    labels_map = {}
    with open(node_labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading node labels"):
            nid = int(row['node_id'])
            label = row['label']  # e.g. 'sybil', 'benign', ...
            labels_map[nid] = label
    return labels_map


###################################################################
# 3) Chunked Edge Reader
###################################################################
def edge_chunk_reader(edges_path, chunk_size=10000):
    """
    A generator that yields chunks (lists) of edges from edges.csv
    where each row looks like:
       source, target, attributes
    and attributes is a string like "{'edge_type': 'xyz'}".
    """
    batch = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row['source'])
            dst = int(row['target'])

            attr_dict = ast.literal_eval(row['attributes'])
            edge_type = attr_dict.get('edge_type', 'unknown')

            batch.append((src, dst, edge_type))

            if len(batch) == chunk_size:
                yield batch
                batch = []

    # Yield leftover
    if batch:
        yield batch


###################################################################
# 4) On-Demand Embedding Loader (One CSV Column with Comma-Separated Values)
###################################################################
def load_node_embeddings_for_batch(node_ids, embedding_offset_map, embed_path, embedding_dim):
    """
    Loads embeddings for the given node_ids from 'embed_path'.
    The file has lines like: node_id, "val1,val2,val3,...,valN"
    
    node_ids: list of node IDs we need
    embedding_offset_map: node_id -> line index (0-based, after the header)
    embed_path: path to CSV (no header or 1-line header we skip)
    embedding_dim: how many floats to parse
    
    Returns: dict { node_id: np.array([...], dtype=np.float32) }
    """
    node_embeddings = {}

    # Build a line_offset -> node_id dict
    needed_offsets = {embedding_offset_map[nid]: nid for nid in node_ids}
    sorted_offsets = sorted(needed_offsets.keys())
    if not sorted_offsets:
        return node_embeddings  # nothing to load

    with open(embed_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip header if present
        next(reader, None)

        offset_idx = 0
        target_offset = sorted_offsets[offset_idx]

        for line_idx, row_vals in enumerate(reader):
            if line_idx < target_offset:
                continue
            elif line_idx > target_offset:
                # Move to next offset
                offset_idx += 1
                if offset_idx >= len(sorted_offsets):
                    break
                target_offset = sorted_offsets[offset_idx]
                if line_idx < target_offset:
                    continue

            # Now line_idx == target_offset
            row_node_id = int(row_vals[0])
            embedding_str = row_vals[1]
            emb_list = [float(x.strip()) for x in embedding_str.split(",")[:embedding_dim]]
            emb_array = np.array(emb_list, dtype=np.float32)
            node_embeddings[row_node_id] = emb_array

            # Advance offset_idx
            offset_idx += 1
            if offset_idx >= len(sorted_offsets):
                break
            target_offset = sorted_offsets[offset_idx]

    return node_embeddings


###################################################################
# 5) Build a Mini-Batch Subgraph
###################################################################
def build_subgraph(chunk_edges, node_type_map):
    """
    Builds adjacency info for a mini-batch of edges.
    Returns:
      edge_dict: dict { (src_type, edge_type, dst_type): [(src, dst), ...], ... }
      unique_nodes: set of all node IDs in this batch
    """
    from collections import defaultdict
    edge_dict = defaultdict(list)
    unique_nodes = set()

    for (src, dst, etype) in chunk_edges:
        s_type = node_type_map[src]
        d_type = node_type_map[dst]
        edge_dict[(s_type, etype, d_type)].append((src, dst))
        unique_nodes.add(src)
        unique_nodes.add(dst)

    return edge_dict, unique_nodes


###################################################################
# 6) Simple Hetero GAT Model (Toy)
###################################################################
class SimpleHeteroGAT(nn.Module):
    """
    A toy multi-edge GAT-like model.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)

        self.attn_l = nn.ParameterDict()
        self.attn_r = nn.ParameterDict()
        self.relation_keys = set()
        self.relations_inited = False

        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def init_relations(self, edge_dict):
        # Initialize attention parameters per relation
        if not self.relations_inited:
            for rel_key in edge_dict.keys():
                rel_str = str(rel_key)
                self.attn_l[rel_str] = nn.Parameter(torch.randn(1))
                self.attn_r[rel_str] = nn.Parameter(torch.randn(1))
            self.relations_inited = True

    def forward(self, x_dict, edge_dict):
        # 1) Project embeddings
        for ntype in x_dict:
            x_dict[ntype] = self.proj(x_dict[ntype])  # [num_local_nodes, hidden_dim]

        # 2) Accumulate messages
        out_dict = {}
        for ntype in x_dict:
            out_dict[ntype] = torch.zeros_like(x_dict[ntype])

        for rel_key, edges in edge_dict.items():
            rel_str = str(rel_key)
            alpha_l = self.attn_l[rel_str]
            alpha_r = self.attn_r[rel_str]
            (src_t, _, dst_t) = rel_key

            for (src, dst) in edges:
                # src, dst are local indices here!
                message = x_dict[src_t][src] * alpha_l + x_dict[dst_t][dst] * alpha_r
                out_dict[dst_t][dst] += message

        # 3) Non-linear activation
        for ntype in out_dict:
            out_dict[ntype] = F.elu(out_dict[ntype])

        # 4) Output layer for 'user'
        if 'user' in out_dict:
            out_dict['user'] = self.out_layer(out_dict['user'])

        return out_dict


###################################################################
# 7) Training Loop with Chunking (Optimized)
###################################################################
def train_in_chunks(
    node_info_path,
    node_labels_path,
    embed_path,
    edges_path,
    embedding_dim=128,
    chunk_size=5000,
    num_epochs=1
):
    """
    Processes the graph in batches of edges, loads embeddings only for needed nodes,
    builds a subgraph, and trains a toy heterogeneous GNN.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build minimal index
    node_type_map, embedding_offset_map = build_node_index(node_info_path, embed_path)
    labels_map = load_labels(node_labels_path)
    print("step 1 done")

    # 2) Label encoding
    unique_labels = sorted(set(labels_map.values()))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    print("step 2 done")

    # 3) Create the model
    model = SimpleHeteroGAT(
        in_dim=embedding_dim,
        hidden_dim=64,
        out_dim=len(label_to_idx)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    print("step 3 done")


    # Main training loop
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        edge_generator = edge_chunk_reader(edges_path, chunk_size=chunk_size)
        for batch_i, chunk_edges in tqdm(enumerate(edge_generator, start=1)):
            # A) Build subgraph adjacency
            edge_dict, unique_nodes = build_subgraph(chunk_edges, node_type_map)

            # B) Group nodes by type
            node_lists = defaultdict(list)
            for gid in unique_nodes:
                ntype = node_type_map[gid]
                node_lists[ntype].append(gid)
            # Sort node IDs
            for ntype in node_lists:
                node_lists[ntype].sort()

            # C) Build local mappings and load embeddings
            #    We'll store x_dict[ntype] with shape [num_local_nodes, embedding_dim]
            local_id_map = {}
            x_dict = {}
            for ntype, g_list in node_lists.items():
                # local reindex: global -> local
                g_to_l = {g: i for i, g in enumerate(g_list)}
                local_id_map[ntype] = g_to_l

                # Load embeddings in one shot
                embeddings = load_node_embeddings_for_batch(
                    node_ids=g_list,
                    embedding_offset_map=embedding_offset_map,
                    embed_path=embed_path,
                    embedding_dim=embedding_dim
                )
                # Create a single NumPy array
                arr_size = (len(g_list), embedding_dim)
                big_np = np.empty(arr_size, dtype=np.float32)
                for i, g in enumerate(g_list):
                    big_np[i] = embeddings[g]

                # Convert to torch tensor
                emb_tensor = torch.from_numpy(big_np).to(device)
                x_dict[ntype] = emb_tensor

            # D) Convert edges to local indices
            local_edge_dict = defaultdict(list)
            for (src_t, e_t, dst_t), e_list in edge_dict.items():
                s_map = local_id_map[src_t]
                d_map = local_id_map[dst_t]
                for (src, dst) in e_list:
                    if src in s_map and dst in d_map:
                        s_local = s_map[src]
                        d_local = d_map[dst]
                        local_edge_dict[(src_t, e_t, dst_t)].append((s_local, d_local))

            # E) Initialize relation keys in model if needed
            model.init_relations(local_edge_dict)

            # F) Forward pass
            out_dict = model(x_dict, local_edge_dict)

            # G) Compute loss on labeled 'user' nodes
            loss = torch.tensor(0.0, device=device)
            num_labeled = 0
            if 'user' in node_lists:
                user_out = out_dict['user']  # [num_local_users, out_dim]
                u_map = local_id_map['user']
                labeled_indices = []
                labeled_values = []
                for g in node_lists['user']:
                    if g in labels_map:
                        labeled_indices.append(u_map[g])
                        labeled_values.append(label_to_idx[labels_map[g]])

                if labeled_indices:
                    labeled_indices_t = torch.tensor(labeled_indices, dtype=torch.long, device=device)
                    labeled_values_t = torch.tensor(labeled_values, dtype=torch.long, device=device)
                    logits = user_out[labeled_indices_t]
                    loss = loss_fn(logits, labeled_values_t)
                    num_labeled = len(labeled_indices)

            # H) Backprop
            if num_labeled > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"  Batch {batch_i}: edges={len(chunk_edges)}, labeled_users={num_labeled}, loss={loss.item():.4f}")

    return model


###############################################################################
# Example: How to run (adjust paths/dimensions to your actual data)
###############################################################################

if __name__ == "__main__":
    node_info_path   = "node_information.csv"  # 'node_id', 'old_id', 'attributes'
    node_labels_path = "node_labels.csv"       # 'node_id', 'label'
    embed_path       = "node_embeddings.csv"   # 'node_id', 'embedding_string'
    edges_path       = "edges.csv"             # 'source', 'target', 'attributes'

    trained_model = train_in_chunks(
        node_info_path=node_info_path,
        node_labels_path=node_labels_path,
        embed_path=embed_path,
        edges_path=edges_path,
        embedding_dim=32,   # match the actual number of floats per node
        chunk_size=5000,
        num_epochs=2
    )

    # Optionally, save the trained model:
    # torch.save(trained_model.state_dict(), "hetero_gnn_model.pth")
