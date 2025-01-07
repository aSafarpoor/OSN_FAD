import csv
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

###############################################################################
# 1) BUILD NODE INDEX
###############################################################################
def build_node_index(node_info_path, embed_path):
    """
    Reads:
      - node_information.csv: [node_id, old_id, attributes]
          where 'attributes' is a JSON-like string: e.g. {'node_type':'user'}
      - node_embeddings.csv: [node_id, embedding_str]
          We only read the line offsets here.

    Returns:
      node_type_map: {node_id -> node_type}
      embedding_offset_map: {node_id -> line_offset_in_embed_file}
    """
    node_type_map = {}
    embedding_offset_map = {}

    # Read node_information.csv to get node_type
    with open(node_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading node info"):
            nid = int(row['node_id'])
            attr_dict = ast.literal_eval(row['attributes'])  # e.g. {'node_type': 'user'}
            node_type = attr_dict.get('node_type', 'unknown')
            node_type_map[nid] = node_type

    # Build offset map from embed_path
    line_num = 0
    with open(embed_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading embedding offsets"):
            nid = int(row['node_id'])
            embedding_offset_map[nid] = line_num
            line_num += 1

    return node_type_map, embedding_offset_map

###############################################################################
# 2) LOAD NODE LABELS
###############################################################################
def load_labels(node_labels_path):
    """
    Reads node_labels.csv with columns: [node_id, label].
    Returns: labels_map {node_id -> label_str}
    """
    labels_map = {}
    with open(node_labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading node labels"):
            nid = int(row['node_id'])
            label_str = row['label']
            labels_map[nid] = label_str
    return labels_map

###############################################################################
# 3) EDGE CHUNK READER
###############################################################################
def edge_chunk_reader(edges_path, chunk_size=10000):
    """
    Reads edges.csv in small batches.
    CSV columns: source, target, attributes
      where attributes is JSON-like, e.g. {'edge_type':'xyz'}

    Yields lists of (src, dst, edge_type) up to chunk_size at a time.
    """
    batch = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row['source'])
            dst = int(row['target'])
            attr_dict = ast.literal_eval(row['attributes'])
            e_type = attr_dict.get('edge_type', 'unknown')

            batch.append((src, dst, e_type))
            if len(batch) == chunk_size:
                yield batch
                batch = []

    if batch:
        yield batch

###############################################################################
# 4) LOAD EMBEDDINGS ON DEMAND
###############################################################################
def load_node_embeddings_for_batch(node_ids, embedding_offset_map, embed_path, embedding_dim):
    """
    Loads embeddings for the given node_ids from a CSV with lines like:
      node_id, "val1,val2,val3,..."
    The 'embedding_offset_map' says which line index each node_id is on.

    Returns: { node_id -> np.array([...], dtype=float32) }
    """
    node_embeddings = {}
    if not node_ids:
        return node_embeddings  # empty

    # Build a lookup from line_offset -> node_id
    needed_offsets = {embedding_offset_map[nid]: nid for nid in node_ids}
    sorted_offsets = sorted(needed_offsets.keys())

    with open(embed_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # skip header (if any)
        next(reader, None)

        offset_idx = 0
        target_offset = sorted_offsets[offset_idx]

        for line_idx, row_vals in enumerate(reader):
            if line_idx < target_offset:
                continue
            elif line_idx > target_offset:
                offset_idx += 1
                if offset_idx >= len(sorted_offsets):
                    break
                target_offset = sorted_offsets[offset_idx]
                if line_idx < target_offset:
                    continue

            # line_idx == target_offset
            row_node_id = int(row_vals[0])
            embedding_str = row_vals[1]  # e.g. "0.1,0.2,0.3,..."
            emb_list = [float(x.strip()) for x in embedding_str.split(",")[:embedding_dim]]
            emb_array = np.array(emb_list, dtype=np.float32)
            node_embeddings[row_node_id] = emb_array

            offset_idx += 1
            if offset_idx >= len(sorted_offsets):
                break
            target_offset = sorted_offsets[offset_idx]

    return node_embeddings

###############################################################################
# 5) BUILD SUBGRAPH
###############################################################################
def build_subgraph(chunk_edges, node_type_map):
    """
    chunk_edges: list of (src, dst, edge_type)
    node_type_map: {node_id -> node_type}
    Returns:
      edge_dict: { (src_type,e_type,dst_type) : [(src_id, dst_id), ...], ... }
      unique_nodes: set of node_ids in this batch
    """
    edge_dict = defaultdict(list)
    unique_nodes = set()

    for (src, dst, e_type) in chunk_edges:
        stype = node_type_map[src]
        dtype = node_type_map[dst]
        edge_dict[(stype, e_type, dtype)].append((src, dst))
        unique_nodes.add(src)
        unique_nodes.add(dst)

    return edge_dict, unique_nodes

###############################################################################
# 6) SIMPLE HETERO GAT MODEL
###############################################################################
class SimpleHeteroGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.attn_l = nn.ParameterDict().to(self.device)
        self.attn_r = nn.ParameterDict().to(self.device)
        self.relations_inited = False
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def init_relations(self, edge_dict):
        """Initialize attention parameters once per unique relation."""
        if not self.relations_inited:
            for rel_key in edge_dict.keys():
                r_str = str(rel_key)
                self.attn_l[r_str] = nn.Parameter(torch.randn(1)).to(self.device)
                self.attn_r[r_str] = nn.Parameter(torch.randn(1)).to(self.device)
            self.relations_inited = True

    def forward(self, x_dict, edge_dict):
        """
        x_dict: { ntype: Tensor of shape [num_local_nodes, in_dim], all on same device }
        edge_dict: { (src_type,e_type,dst_type): [(src_local, dst_local), ...], ... }
        """

        

 
        # 1) Project
        for ntype in x_dict:
            x_dict[ntype] = self.proj(x_dict[ntype])  # [N, hidden_dim]

        # 2) Accumulate messages
        out_dict = {ntype: torch.zeros_like(x_dict[ntype]) for ntype in x_dict}

        for rel_key, edges in edge_dict.items():
            r_str = str(rel_key)
            alpha_l = self.attn_l[r_str]  # shape [1]
            alpha_r = self.attn_r[r_str]
            (src_t, _, dst_t) = rel_key

            for (src_loc, dst_loc) in edges:
                # 'src_loc' indexes row in x_dict[src_t]
                # 'dst_loc' indexes row in x_dict[dst_t]
                # x_dict[src_t] = x_dict[src_t].to(device)
                # x_dict[dst_t] = x_dict[dst_t].to(device)
                # print("Device of tensor:", x_dict[src_t].device)
                # print("Device of tensor:", x_dict[src_t][src_loc].device)
                # print("Device of tensor:", alpha_l.device)

                message = x_dict[src_t][src_loc] * alpha_l + x_dict[dst_t][dst_loc] * alpha_r
                out_dict[dst_t][dst_loc] += message

        # 3) Non-linear activation
        for ntype in out_dict:
            out_dict[ntype] = F.elu(out_dict[ntype])

        # 4) If 'user' in out_dict, apply final layer
        if 'user' in out_dict:
            out_dict['user'] = self.out_layer(out_dict['user'])

        return out_dict

###############################################################################
# 7) TRAIN IN CHUNKS
###############################################################################
def train_in_chunks(
    node_info_path,
    node_labels_path,
    embed_path,
    edges_path,
    embedding_dim=32,
    chunk_size=10000,
    num_epochs=1
):
    """
    Main training function that:
      - Loads node types/labels
      - Reads edges in chunks
      - Builds a subgraph per chunk
      - Runs a simple HeteroGAT model
      - Ensures all data is on the correct device to avoid mismatch
    """
    ###########################################################################
    # A) DEVICE SETUP
    ###########################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    print("step A done")

    ###########################################################################
    # B) LOAD MAPS + LABELS
    ###########################################################################
    node_type_map, embedding_offset_map = build_node_index(node_info_path, embed_path)
    labels_map = load_labels(node_labels_path)

    # Label encoding
    unique_labels = sorted(set(labels_map.values()))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    num_classes = len(label_to_idx)
    print("step B done")

    ###########################################################################
    # C) CREATE MODEL
    ###########################################################################
    model = SimpleHeteroGAT(
        in_dim=embedding_dim,
        hidden_dim=64,
        out_dim=num_classes
    ).to(device)  # move model to device

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    print("step C done")

    ###########################################################################
    # D) TRAIN LOOP
    ###########################################################################
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        edge_generator = edge_chunk_reader(edges_path, chunk_size=chunk_size)
        C = 0
        for batch_i, chunk_edges in tqdm(enumerate(edge_generator, start=1)):
            C +=1
            if C>2:
                break
            ###############################################################
            # 1) BUILD SUBGRAPH
            ###############################################################
            edge_dict, unique_nodes = build_subgraph(chunk_edges, node_type_map)

            ###############################################################
            # 2) GROUP NODES BY TYPE + CREATE LOCAL ID MAP
            ###############################################################
            node_lists = defaultdict(list)
            for gid in unique_nodes:
                ntype = node_type_map[gid]
                node_lists[ntype].append(gid)
            for ntype in node_lists:
                node_lists[ntype].sort()

            local_id_map = {}
            x_dict = {}

            ###############################################################
            # 3) LOAD EMBEDDINGS + BUILD x_dict
            ###############################################################
            for ntype, g_list in node_lists.items():
                # local reindex
                g_to_l = {g: i for i, g in enumerate(g_list)}
                local_id_map[ntype] = g_to_l

                # load embeddings from CSV
                emb_data = load_node_embeddings_for_batch(
                    node_ids=g_list,
                    embedding_offset_map=embedding_offset_map,
                    embed_path=embed_path,
                    embedding_dim=embedding_dim
                )

                # combine into one array
                arr_size = (len(g_list), embedding_dim)
                big_np = np.zeros(arr_size, dtype=np.float32)
                for i, g in enumerate(g_list):
                    big_np[i] = emb_data[g]

                # convert to torch on device
                emb_tensor = torch.from_numpy(big_np).to(device)
                x_dict[ntype] = emb_tensor

            ###############################################################
            # 4) CONVERT EDGES TO LOCAL INDICES
            ###############################################################
            local_edge_dict = defaultdict(list)
            for (src_t, e_type, dst_t), e_list in edge_dict.items():
                s_map = local_id_map[src_t]
                d_map = local_id_map[dst_t]
                for (src, dst) in e_list:
                    if src in s_map and dst in d_map:
                        s_local = s_map[src]
                        d_local = d_map[dst]
                        local_edge_dict[(src_t, e_type, dst_t)].append((s_local, d_local))

            ###############################################################
            # 5) MODEL INIT RELATIONS + FORWARD
            ###############################################################
            model.init_relations(local_edge_dict)
            out_dict = model(x_dict, local_edge_dict)  # all on same device

            ###############################################################
            # 6) LOSS COMPUTATION (on 'user' nodes)
            ###############################################################
            loss = torch.tensor(0.0, device=device)
            num_labeled = 0

            if 'user' in node_lists:
                user_out = out_dict['user']  # shape [num_local_users, out_dim]
                user_map = local_id_map['user']
                labeled_indices = []
                labeled_values = []
                for g in node_lists['user']:
                    if g in labels_map:
                        labeled_indices.append(user_map[g])
                        labeled_values.append(label_to_idx[labels_map[g]])

                if labeled_indices:
                    labeled_indices_t = torch.tensor(labeled_indices, dtype=torch.long, device=device)
                    labeled_values_t = torch.tensor(labeled_values, dtype=torch.long, device=device)
                    logits = user_out[labeled_indices_t]
                    loss = loss_fn(logits, labeled_values_t)
                    num_labeled = len(labeled_indices)

            ###############################################################
            # 7) OPTIMIZER STEP
            ###############################################################
            if num_labeled > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"  Batch {batch_i}: edges={len(chunk_edges)}, labeled_users={num_labeled}, loss={loss.item():.4f}")

    return model


###############################################################################
# 8) RUN / TEST
###############################################################################
if __name__ == "__main__":
    # Adjust CSV file paths here
    node_info_path   = "node_information.csv"
    node_labels_path = "node_labels.csv"
    embed_path       = "node_embeddings.csv"
    edges_path       = "edges.csv"

    trained_model = train_in_chunks(
        node_info_path=node_info_path,
        node_labels_path=node_labels_path,
        embed_path=embed_path,
        edges_path=edges_path,
        embedding_dim=32,   # Change to match your embedding vector size
        chunk_size=50000,
        num_epochs=2
    )

    # Optionally save the final model
    torch.save(trained_model.state_dict(), "hetero_gnn_model.pth")
