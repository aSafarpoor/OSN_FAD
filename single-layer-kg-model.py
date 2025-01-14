import os
import csv
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

import random
import datetime
import logging

# PyG imports
from torch_geometric.nn import GATConv, HeteroConv

###############################################################################
# 1) SETUP LOGGING
###############################################################################
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Configure logging: writes to console + a timestamped file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'my_log_file_{timestamp}.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

###############################################################################
# 2) UTILS FOR REPRODUCIBILITY + FILE I/O
###############################################################################
def set_random_seeds(seed=42):
    """
    Make experiment reproducible by setting seeds for Python, NumPy, and Torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _read_ids(filename):
    """
    Read a list of node IDs from a text file, one ID per line.
    """
    with open(filename, 'r') as f:
        lines = f.read().strip().split()
    return [int(x) for x in lines]

def _write_ids(filename, ids_set):
    """
    Write a set or list of node IDs to a text file, one ID per line.
    """
    with open(filename, 'w') as f:
        for val in sorted(ids_set):
            f.write(f"{val}\n")

###############################################################################
# 3) LOAD CSVs
###############################################################################
def build_node_type_map(node_info_path):
    """
    Reads node_information.csv: [node_id, old_id, attributes]
      'attributes' is a JSON-like string, e.g. {"node_type": "user"}
    Returns: 
      node_type_map = {node_id -> node_type}
    """
    node_type_map = {}
    with open(node_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading node info (once)"):
            nid = int(row['node_id'])
            attr_dict = ast.literal_eval(row['attributes'])
            node_type = attr_dict.get('node_type', 'unknown')
            node_type_map[nid] = node_type
    return node_type_map

def load_labels(node_labels_path):
    """
    Reads node_labels.csv: [node_id, label].
    Returns: labels_map = {node_id -> label_str}
    """
    labels_map = {}
    with open(node_labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading node labels (once)"):
            nid = int(row['node_id'])
            label_str = row['label']
            labels_map[nid] = label_str
    return labels_map

def load_all_embeddings(embed_path, embedding_dim):
    """
    Reads ALL node_embeddings.csv once.
      Format assumed: node_id, "val1,val2,val3,..."
    Returns:
      all_embeddings = { node_id -> np.array([embedding_dim], dtype=float32) }
    """
    all_embeddings = {}
    with open(embed_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Attempt to skip header if present
        header = next(reader, None)
        for row_vals in tqdm(reader, desc="Reading all embeddings (once)"):
            if len(row_vals) < 2:
                continue
            row_node_id = int(row_vals[0])
            embedding_str = row_vals[1]
            emb_list = [float(x.strip()) for x in embedding_str.split(",")[:embedding_dim]]
            emb_array = np.array(emb_list, dtype=np.float32)
            all_embeddings[row_node_id] = emb_array
    return all_embeddings

def load_all_edges(edges_path):
    """
    Reads ALL edges.csv once into a list of (src, dst, edge_type).
      edges.csv columns: source, target, attributes
         'attributes' is a JSON-like string, e.g. {"edge_type": "xyz"}
    Returns:
      edges_list = [ (src, dst, e_type), ... ]
    """
    edges_list = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading edges (once)"):
            src = int(row['source'])
            dst = int(row['target'])
            attr_dict = ast.literal_eval(row['attributes'])
            e_type = attr_dict.get('edge_type', 'unknown')
            edges_list.append((src, dst, e_type))
    return edges_list

###############################################################################
# 4) DEGREE-BASED SPLITTING INTO TEST1, TEST2, TEST3, TEST4
###############################################################################
def compute_user_degrees_for_benign_sybil_users(edges_list, node_type_map, labels_map):
    """
    For each user node labeled benign/sybil, compute:
      - total_degree
      - user_deg (# of neighbors that are 'user')
      - tweet_deg (# of neighbors that are 'tweet')
    Returns:
      (user_ids, deg_map, user_deg_map, tweet_deg_map)
    """
    valid_user_nodes = set()
    for nid, lab in labels_map.items():
        if node_type_map.get(nid) == 'user' and lab in ['benign', 'sybil']:
            valid_user_nodes.add(nid)

    deg_map = defaultdict(int)
    user_deg_map = defaultdict(int)
    tweet_deg_map = defaultdict(int)

    for (src, dst, e_type) in edges_list:
        src_type = node_type_map[src]
        dst_type = node_type_map[dst]

        # If src is a valid user
        if src in valid_user_nodes:
            deg_map[src] += 1
            if dst_type == 'user':
                user_deg_map[src] += 1
            elif dst_type == 'tweet':
                tweet_deg_map[src] += 1

        # If dst is a valid user
        if dst in valid_user_nodes:
            deg_map[dst] += 1
            if src_type == 'user':
                user_deg_map[dst] += 1
            elif src_type == 'tweet':
                tweet_deg_map[dst] += 1

    # Fill missing zeros for user nodes that never appeared in edges
    for u in valid_user_nodes:
        deg_map.setdefault(u, 0)
        user_deg_map.setdefault(u, 0)
        tweet_deg_map.setdefault(u, 0)

    return (
        list(valid_user_nodes),
        dict(deg_map),
        dict(user_deg_map),
        dict(tweet_deg_map),
    )

def split_into_4_test_sets(
    user_ids,
    labels_map,
    deg_map,
    user_deg_map,
    tweet_deg_map,
    test_size_each=100,
    test4_ratio=0.4,
    seed=42,
    save_dir='.'
):
    """
    Splits user_ids into:
      test1: 100 nodes with least user_deg
      test2: 100 nodes with least tweet_deg
      test3: 100 nodes with least (user_deg+tweet_deg)
      test4: 40% of REMAINING + test1 + test2 + test3
      train: everything else

    Writes these sets to disk if not already present; otherwise loads them.
    Returns (test1, test2, test3, test4, train_set).
    """
    f_test1 = os.path.join(save_dir, 'testset1.txt')
    f_test2 = os.path.join(save_dir, 'testset2.txt')
    f_test3 = os.path.join(save_dir, 'testset3.txt')
    f_test4 = os.path.join(save_dir, 'testset4.txt')
    f_train = os.path.join(save_dir, 'trainset.txt')

    # If all exist, read and return
    if all(os.path.exists(f) for f in [f_test1, f_test2, f_test3, f_test4, f_train]):
        print("All test/train set files found. Reading from disk...")
        test1 = set(_read_ids(f_test1))
        test2 = set(_read_ids(f_test2))
        test3 = set(_read_ids(f_test3))
        test4 = set(_read_ids(f_test4))
        train_set = set(_read_ids(f_train))
        return test1, test2, test3, test4, train_set

    print("Creating new test/train splits...")
    set_random_seeds(seed)

    # Only user nodes that are labeled benign/sybil
    candidate_nodes = []
    for uid in user_ids:
        lab = labels_map.get(uid)
        if lab in ['benign', 'sybil']:
            candidate_nodes.append(uid)

    # test1: 100 least user_deg
    test1_sorted = sorted(candidate_nodes, key=lambda x: user_deg_map.get(x, 0))
    test1 = set(test1_sorted[:test_size_each])

    # test2: 100 least tweet_deg
    test2_sorted = sorted(candidate_nodes, key=lambda x: tweet_deg_map.get(x, 0))
    test2 = set(test2_sorted[:test_size_each])

    # test3: 100 least (user_deg + tweet_deg)
    test3_sorted = sorted(candidate_nodes, key=lambda x: (user_deg_map[x] + tweet_deg_map[x]))
    test3 = set(test3_sorted[:test_size_each])

    combined_123 = test1.union(test2).union(test3)

    # test4: 40% of the remain + test1+test2+test3
    remain = [u for u in candidate_nodes if u not in combined_123]
    random.shuffle(remain)
    test4_size = int(len(remain) * test4_ratio)
    test4_part = remain[:test4_size]
    test4 = set(test4_part).union(combined_123)

    train_set = set(candidate_nodes) - test4

    # Write to disk
    _write_ids(f_test1, test1)
    _write_ids(f_test2, test2)
    _write_ids(f_test3, test3)
    _write_ids(f_test4, test4)
    _write_ids(f_train, train_set)

    return test1, test2, test3, test4, train_set

def print_dataset_stats(name, node_ids, deg_map, user_deg_map, tweet_deg_map):
    """
    Print average degree, user-degree, tweet-degree for the given set of node_ids.
    """
    if not node_ids:
        print(f"[{name}] is empty.")
        return
    deg_vals = [deg_map[n] for n in node_ids]
    user_vals = [user_deg_map[n] for n in node_ids]
    tweet_vals = [tweet_deg_map[n] for n in node_ids]
    print(
        f"[{name}] count={len(node_ids):5d}, "
        f"avgDeg={np.mean(deg_vals):.2f}, "
        f"avgUserDeg={np.mean(user_vals):.2f}, "
        f"avgTweetDeg={np.mean(tweet_vals):.2f}"
    )

###############################################################################
# 5) CHUNKER (in-memory)
###############################################################################
def chunk_edges_in_memory(edges_list, chunk_size=100000):
    """
    Given a list of edges in memory, yield smaller chunks.
    """
    for i in range(0, len(edges_list), chunk_size):
        yield edges_list[i:i + chunk_size]

###############################################################################
# 6) BUILD SUBGRAPH
###############################################################################
def build_subgraph(chunk_edges, node_type_map):
    """
    chunk_edges: list of (src, dst, e_type)
    node_type_map: {node_id -> node_type}
    Returns:
      edge_dict = { (stype,e_type,dtype) : [(src_id, dst_id), ...], ... }
      unique_nodes = set of node_ids in this chunk
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

def build_hetero_data(x_dict, edge_dict):
    """
    Converts (x_dict, edge_dict) into a PyG HeteroData object.
      x_dict: { node_type: FloatTensor [num_nodes, in_dim] }
      edge_dict: { (src_type,e_type,dst_type): [(src_idx, dst_idx), ...], ... }
    """
    from torch_geometric.data import HeteroData
    data = HeteroData()
    # 1) Node features
    for node_type, x_tensor in x_dict.items():
        data[node_type].x = x_tensor
    # 2) Edges
    for (src_type, e_type, dst_type), edges in edge_dict.items():
        if len(edges) > 0:
            src, dst = zip(*edges)
        else:
            src, dst = [], []
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        data[(src_type, dst_type)].edge_index = torch.stack([src, dst], dim=0)
    return data

###############################################################################
# 7) SIMPLE HETERO GAT (PyG)
###############################################################################
class SimpleHeteroGATPyG(nn.Module):
    """
    A refactored heterogeneous multi-head GAT using PyTorch Geometric.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'[SimpleHeteroGATPyG] Using device: {self.device}')
        print(f'[SimpleHeteroGATPyG] Using device: {self.device}')  # also print to console

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.proj = nn.Linear(in_dim, hidden_dim)
        self.hetero_conv = None  # Will build in forward() if needed
        self.out_layer = nn.Linear(hidden_dim * num_heads, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x_dict, edge_dict):
        data = build_hetero_data(x_dict, edge_dict).to(self.device)

        # Project each node_type
        for node_type in data.node_types:
            x_in = data[node_type].x
            data[node_type].x = self.proj(x_in)  # [num_nodes, hidden_dim]

        # Build HeteroConv if not done yet
        if self.hetero_conv is None:
            conv_dict = {}
            for edge_type in data.edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=self.num_heads,
                    concat=True,
                    negative_slope=0.2,
                    dropout=0.3,
                    add_self_loops=False
                )
            self.hetero_conv = HeteroConv(conv_dict, aggr='sum').to(self.device)

        out_feats = self.hetero_conv(
            x_dict={ntype: data[ntype].x for ntype in data.node_types},
            edge_index_dict={etype: data[etype].edge_index for etype in data.edge_types}
        )
        # Apply ELU
        for ntype in out_feats:
            out_feats[ntype] = F.elu(out_feats[ntype])
        # If 'user' is present, apply the final out_layer
        if 'user' in out_feats:
            out_feats['user'] = self.out_layer(out_feats['user'])
        return out_feats

###############################################################################
# 8) METRICS
###############################################################################


def compute_metrics(logits, labels_t, loss_fn):
    """
    logits : Tensor [N, 2] for binary classification
    labels_t: Tensor [N] with {0,1}
    Returns { 'loss', 'acc', 'auc', 'prec', 'rec', 'f1' }
    """
    # Compute loss
    loss_val = loss_fn(logits, labels_t).item()
    
    # Get predicted class (0 or 1)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    # Convert labels and probabilities to numpy
    labels_np = labels_t.cpu().numpy()
    
    # Accuracy
    acc_val = accuracy_score(labels_np, preds)
    
    # AUC
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    if len(set(labels_np)) > 1:
        auc_val = roc_auc_score(labels_np, probs)
    else:
        auc_val = 0.5  # Fallback if only one class is present
    
    # Precision, Recall, F1
    prec_val = precision_score(labels_np, preds, pos_label=1, zero_division=0)
    rec_val  = recall_score(labels_np, preds, pos_label=1, zero_division=0)
    f1_val   = f1_score(labels_np, preds, pos_label=1, zero_division=0)
    
    return {
        'loss': loss_val,
        'acc': acc_val,
        'auc': auc_val,
        'prec': prec_val,
        'rec': rec_val,
        'f1': f1_val
    }


###############################################################################
# 9) TRAIN IN CHUNKS, EVALUATE ON TEST1/TEST2/TEST3/TEST4
###############################################################################
def train_in_chunks(
    node_info_path,
    node_labels_path,
    embed_path,
    edges_path,
    embedding_dim=32,
    chunk_size=10000,
    num_epochs=1,
    test_size_each=100,
    test4_ratio=0.4,
    seed=42
):
    """
    Main function that:
      - Loads node types, labels, embeddings, edges (once).
      - Splits user nodes with sybil/benign into test1, test2, test3, test4, and train.
      - Builds subgraph per chunk and trains a Hetero GAT.
      - Prints/logs test metrics (for each test set) each epoch.
    """
    # 1) Set seeds
    set_random_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device: {device}")
    print(f"Running on device: {device}")

    # 2) LOAD EVERYTHING (ONCE)
    node_type_map = build_node_type_map(node_info_path)
    labels_map = load_labels(node_labels_path)
    all_embeddings = load_all_embeddings(embed_path, embedding_dim)
    edges_list = load_all_edges(edges_path)

    # 3) Compute degrees => split => test1, test2, test3, test4, train_set
    user_ids, deg_map, user_deg_map, tweet_deg_map = compute_user_degrees_for_benign_sybil_users(
        edges_list, node_type_map, labels_map
    )

    test1, test2, test3, test4, train_set = split_into_4_test_sets(
        user_ids      = user_ids,
        labels_map    = labels_map,
        deg_map       = deg_map,
        user_deg_map  = user_deg_map,
        tweet_deg_map = tweet_deg_map,
        test_size_each= test_size_each,
        test4_ratio   = test4_ratio,
        seed          = seed,
        save_dir      = '.'
    )

    # (Optional) Print stats about each set
    print_dataset_stats("train",  train_set, deg_map, user_deg_map, tweet_deg_map)
    print_dataset_stats("test1",  test1,     deg_map, user_deg_map, tweet_deg_map)
    print_dataset_stats("test2",  test2,     deg_map, user_deg_map, tweet_deg_map)
    print_dataset_stats("test3",  test3,     deg_map, user_deg_map, tweet_deg_map)
    print_dataset_stats("test4",  test4,     deg_map, user_deg_map, tweet_deg_map)
    all_test = test1.union(test2).union(test3).union(test4)
    print_dataset_stats("all_test", all_test, deg_map, user_deg_map, tweet_deg_map)

    # 4) CREATE MODEL
    num_classes = 2  # sybil=1, benign=0
    model = SimpleHeteroGATPyG(
        in_dim=embedding_dim,
        hidden_dim=64,
        out_dim=num_classes,
        num_heads=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # 5) TRAIN LOOP
    
    for epoch in range(num_epochs):
        logging.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        epoch_train_loss = 0.0

        # We'll track metrics for each test set across the chunked training
        test_metrics_dict = {
            'test1': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'count': 0},
            'test2': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'count': 0},
            'test3': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'count': 0},
            'test4': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'count': 0},
        }
        
        random.seed(epoch)
        random.shuffle(edges_list)


        # We'll chunk edges in memory
        edge_chunks = chunk_edges_in_memory(edges_list, chunk_size=chunk_size)

        for batch_i, chunk_edges_ in enumerate(edge_chunks, start=1):
            # 1) Build subgraph
            edge_dict, unique_nodes = build_subgraph(chunk_edges_, node_type_map)

            # 2) Group by node_type & build x_dict
            node_lists = defaultdict(list)
            for gid in unique_nodes:
                ntype = node_type_map[gid]
                node_lists[ntype].append(gid)
            for ntype in node_lists:
                node_lists[ntype].sort()

            local_id_map = {}
            x_dict = {}
            for ntype, g_list in node_lists.items():
                g_to_local = {g: i for i, g in enumerate(g_list)}
                local_id_map[ntype] = g_to_local

                # Gather embeddings from our in-memory dictionary
                arr_size = (len(g_list), embedding_dim)
                big_np = np.zeros(arr_size, dtype=np.float32)
                for i, g in enumerate(g_list):
                    if g in all_embeddings:
                        big_np[i] = all_embeddings[g]
                x_dict[ntype] = torch.from_numpy(big_np).to(device)

            # 3) Convert edges to local indices
            local_edge_dict = defaultdict(list)
            for (src_t, e_type, dst_t), e_list in edge_dict.items():
                s_map = local_id_map[src_t]
                d_map = local_id_map[dst_t]
                for (src, dst) in e_list:
                    if src in s_map and dst in d_map:
                        s_local = s_map[src]
                        d_local = d_map[dst]
                        local_edge_dict[(src_t, e_type, dst_t)].append((s_local, d_local))

            # Forward pass
            out_dict = model(x_dict, local_edge_dict)

            # 4) Compute loss on train user nodes
            loss = torch.tensor(0.0, device=device)
            num_labeled_train = 0
            if 'user' in node_lists:
                user_out = out_dict['user']
                user_map = local_id_map['user']

                # Gather train user indices
                train_indices, train_labels = [], []
                for g in node_lists['user']:
                    if g in train_set and g in labels_map:
                        lab = labels_map[g]
                        if lab == 'sybil':
                            train_indices.append(user_map[g])
                            train_labels.append(1)
                        elif lab == 'benign':
                            train_indices.append(user_map[g])
                            train_labels.append(0)

                # Train loss
                if train_indices:
                    train_idx_t = torch.tensor(train_indices, dtype=torch.long, device=device)
                    train_lbl_t = torch.tensor(train_labels, dtype=torch.long, device=device)
                    logits_train = user_out[train_idx_t]
                    loss = loss_fn(logits_train, train_lbl_t)
                    num_labeled_train = len(train_indices)

                # Evaluate on each test set
                def evaluate_testset(test_nodes, key):
                    idxs, lbls = [], []
                    for g in node_lists['user']:
                        if g in test_nodes and g in labels_map:
                            lab = labels_map[g]
                            if lab == 'sybil':
                                idxs.append(user_map[g])
                                lbls.append(1)
                            elif lab == 'benign':
                                idxs.append(user_map[g])
                                lbls.append(0)
                    if idxs:
                        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
                        lbls_t = torch.tensor(lbls, dtype=torch.long, device=device)
                        logits_test = user_out[idxs_t]
                        m = compute_metrics(logits_test, lbls_t, loss_fn)
                        test_metrics_dict[key]['loss'] += m['loss']
                        test_metrics_dict[key]['acc']  += m['acc']
                        test_metrics_dict[key]['auc']  += m['auc']
                        test_metrics_dict[key]['prec'] += m['prec']
                        test_metrics_dict[key]['rec']  += m['rec']
                        test_metrics_dict[key]['f1']   += m['f1']
                        test_metrics_dict[key]['count'] += 1


                # Evaluate each test set
                evaluate_testset(test1, 'test1')
                evaluate_testset(test2, 'test2')
                evaluate_testset(test3, 'test3')
                evaluate_testset(test4, 'test4')

            # 5) Backprop if we have labeled data
            if num_labeled_train > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item()

            msg = (f"  Batch {batch_i}: edges={len(chunk_edges_)}, "
                   f"train_users={num_labeled_train}, loss={loss.item():.4f}")
            logging.info(msg)
            print(msg)

        # end of epoch => average metrics
        avg_train_loss = epoch_train_loss / max(batch_i, 1)

        results_str = []
        for k in ['test1', 'test2', 'test3', 'test4']:
            c = test_metrics_dict[k]['count']
            if c > 0:
                test_metrics_dict[k]['loss'] /= c
                test_metrics_dict[k]['acc']  /= c
                test_metrics_dict[k]['auc']  /= c
                test_metrics_dict[k]['prec'] /= c
                test_metrics_dict[k]['rec']  /= c
                test_metrics_dict[k]['f1']   /= c
            
            results_str.append(
                f"{k}: loss={test_metrics_dict[k]['loss']:.4f}, "
                f"acc={test_metrics_dict[k]['acc']:.4f}, "
                f"auc={test_metrics_dict[k]['auc']:.4f}, "
                f"prec={test_metrics_dict[k]['prec']:.4f}, "
                f"rec={test_metrics_dict[k]['rec']:.4f}, "
                f"f1={test_metrics_dict[k]['f1']:.4f}"
            )

        msg_epoch = (
            f"[Epoch {epoch+1}] AvgTrainLoss={avg_train_loss:.4f} | "
            + " | ".join(results_str)
        )
        logging.info(msg_epoch)
        print(msg_epoch)


        # optionally save model checkpoint each epoch
        torch.save(model.state_dict(), "training_time_hetero_gnn_model.pth")

    return model

###############################################################################
# 10) MAIN
###############################################################################
if __name__ == "__main__":
    node_info_path   = "node_information.csv"
    node_labels_path = "node_labels.csv"
    embed_path       = "node_embeddings.csv"
    edges_path       = "edges.csv"

    # Adjust as needed
    chunk_size = 1000000
    num_epochs = 70
    embedding_dim = 32  # must match your embedding dimension

    trained_model = train_in_chunks(
        node_info_path=node_info_path,
        node_labels_path=node_labels_path,
        embed_path=embed_path,
        edges_path=edges_path,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        num_epochs=num_epochs,
        seed=42,         
        test_size_each=100,  # how many go into test1/test2/test3
        test4_ratio=0.4      # fraction of remainder that goes into test4
    )

    checkpoint = {
        'state_dict': trained_model.state_dict(),
        'metadata': {
            'trained_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Hetero-GNN model with extended metadata (4 test splits).',
            'num_epochs': num_epochs,
            'chunk_size': chunk_size
        }
    }
    final_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"hetero_gnn_model_{final_ts}.pth"
    torch.save(checkpoint, filename)
    logging.info(f"Final model saved to {filename}")
    print(f"Final model saved to {filename}")
