import os
import csv
import ast
import random
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score

# PyG imports
from torch_geometric.nn import GATConv, HeteroConv

###############################################################################
# 1) SETUP LOGGING
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('my_log_file.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

###############################################################################
# 2) UTILITY FUNCTIONS
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
# 3) READING CSV FILES
###############################################################################
def build_node_type_map(node_info_path):
    """
    Reads node_information.csv (columns: node_id, old_id, attributes)
      'attributes' is a JSON-like string, e.g. {"node_type": "user"}
    Returns { node_id -> node_type }
    """
    node_type_map = {}
    with open(node_info_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading node_info"):
            nid = int(row['node_id'])
            attr_dict = ast.literal_eval(row['attributes'])
            node_type = attr_dict.get('node_type', 'unknown')
            node_type_map[nid] = node_type
    return node_type_map

def load_labels(node_labels_path):
    """
    Reads node_labels.csv (columns: node_id, label)
    Returns { node_id -> label_str }
    """
    labels_map = {}
    with open(node_labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading node_labels"):
            nid = int(row['node_id'])
            label_str = row['label']
            labels_map[nid] = label_str
    return labels_map

def load_all_embeddings(embed_path, embedding_dim):
    """
    Reads node_embeddings.csv (assumes columns: node_id, embedding_str)
      where embedding_str looks like "val1,val2,val3,..."
    Returns { node_id -> np.array([...], dtype=float32) }
    """
    all_embeddings = {}
    with open(embed_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Attempt to skip header if present
        header = next(reader, None)
        for row_vals in tqdm(reader, desc="Reading embeddings"):
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
    Reads edges.csv (columns: source, target, attributes)
      'attributes' is a JSON-like string, e.g. {"edge_type": "foo"}
    Returns a list of (source, target, edge_type)
    """
    edges_list = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading edges"):
            src = int(row['source'])
            dst = int(row['target'])
            attr_dict = ast.literal_eval(row['attributes'])
            e_type = attr_dict.get('edge_type', 'unknown')
            edges_list.append((src, dst, e_type))
    return edges_list

###############################################################################
# 4) COMPUTE USER DEGREES & SPLIT LOGIC
###############################################################################
def compute_user_degrees_for_benign_sybil_users(edges_list, node_type_map, labels_map):
    """
    For each user node labeled benign/sybil:
      - total_degree (#neighbors of any type)
      - user_deg (# of user neighbors)
      - tweet_deg (# of tweet neighbors)
    Returns:
      user_ids (list of user nodes),
      deg_map       : { u -> total_degree },
      user_deg_map  : { u -> # user neighbors },
      tweet_deg_map : { u -> # tweet neighbors }
    """
    valid_user_nodes = set()
    for nid, lab in labels_map.items():
        if node_type_map.get(nid) == 'user' and lab in ['benign', 'sybil']:
            valid_user_nodes.add(nid)

    deg_map = defaultdict(int)
    user_deg_map = defaultdict(int)
    tweet_deg_map = defaultdict(int)

    # Count degrees only for these user nodes
    for (src, dst, e_type) in edges_list:
        src_type = node_type_map[src]
        dst_type = node_type_map[dst]

        # If src is a valid user:
        if src in valid_user_nodes:
            deg_map[src] += 1
            if dst_type == 'user':
                user_deg_map[src] += 1
            elif dst_type == 'tweet':
                tweet_deg_map[src] += 1

        # If dst is a valid user:
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

    return list(valid_user_nodes), dict(deg_map), dict(user_deg_map), dict(tweet_deg_map)

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
    Splits the user_ids into:
      test1: 100 nodes with least user_deg
      test2: 100 nodes with least tweet_deg
      test3: 100 nodes with least (user_deg + tweet_deg)
      test4: 40% (test4_ratio) of the REMAINING + test1 + test2 + test3
      train: everything else

    If the files (testset1.txt, testset2.txt, testset3.txt, testset4.txt, trainset.txt)
    already exist in save_dir from a previous run, we just read them.
    Otherwise, we compute them anew and save them.
    Returns (test1, test2, test3, test4, train) as sets of node IDs.
    """

    # Filenames
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
        train = set(_read_ids(f_train))
        return test1, test2, test3, test4, train

    print("Creating new test/train splits...")
    set_random_seeds(seed)

    # Filter only benign/sybil from user_ids
    candidate_nodes = []
    for uid in user_ids:
        lab = labels_map.get(uid, None)
        if lab in ['benign', 'sybil']:
            candidate_nodes.append(uid)

    # 1) test1: 100 least user_deg
    test1_sorted = sorted(candidate_nodes, key=lambda x: user_deg_map.get(x, 0))
    test1 = set(test1_sorted[:test_size_each])

    # 2) test2: 100 least tweet_deg
    test2_sorted = sorted(candidate_nodes, key=lambda x: tweet_deg_map.get(x, 0))
    test2 = set(test2_sorted[:test_size_each])

    # 3) test3: 100 least (user_deg + tweet_deg)
    test3_sorted = sorted(candidate_nodes, key=lambda x: (user_deg_map[x] + tweet_deg_map[x]))
    test3 = set(test3_sorted[:test_size_each])

    combined_123 = test1.union(test2).union(test3)

    # 4) test4: 40% of REMAIN (random) + test1, test2, test3
    remain = [u for u in candidate_nodes if u not in combined_123]
    random.shuffle(remain)
    test4_size = int(len(remain) * test4_ratio)
    test4_part = remain[:test4_size]
    test4 = set(test4_part).union(combined_123)

    train = set(candidate_nodes) - test4

    # Write to disk
    _write_ids(f_test1, test1)
    _write_ids(f_test2, test2)
    _write_ids(f_test3, test3)
    _write_ids(f_test4, test4)
    _write_ids(f_train, train)

    return test1, test2, test3, test4, train

###############################################################################
# 5) PRINT DATASET STATS
###############################################################################
def print_dataset_stats(name, node_ids, deg_map, user_deg_map, tweet_deg_map):
    """
    Print average degree, user-degree, tweet-degree for the given node_ids set.
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
# 6) CHUNKER & SUBGRAPH BUILDING
###############################################################################
def chunk_edges_in_memory(edges_list, chunk_size=100000):
    """
    Yields sublists (chunks) of edges_list, each up to `chunk_size`.
    """
    for i in range(0, len(edges_list), chunk_size):
        yield edges_list[i:i + chunk_size]

def build_subgraph(chunk_edges, node_type_map):
    """
    From a list of edges, build:
      edge_dict = {(src_type, e_type, dst_type): [(src, dst), ...], ...}
      unique_nodes = set of all node IDs
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
    Convert (x_dict, edge_dict) into a PyG HeteroData object.
    """
    from torch_geometric.data import HeteroData
    data = HeteroData()
    # Node features
    for node_type, x_tensor in x_dict.items():
        data[node_type].x = x_tensor
    # Edges
    for (src_type, e_type, dst_type), edges in edge_dict.items():
        if edges:
            src, dst = zip(*edges)
        else:
            src, dst = [], []
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        data[(src_type, dst_type)].edge_index = torch.stack([src, dst], dim=0)
    return data

###############################################################################
# 7) MULTI-LAYER HETERO GAT
###############################################################################
class MultiLayerHeteroGAT(nn.Module):
    """
    A heterogeneous multi-head GAT with `k` hidden layers using PyTorch Geometric.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, num_layers=4):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[MultiLayerHeteroGAT] Using device: {self.device}')

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Input projection (from in_dim to hidden_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # We'll store one HeteroConv per hidden layer
        self.hetero_conv_layers = nn.ModuleList([None] * num_layers)

        # Output layer for 'user' node classification
        self.out_layer = nn.Linear(hidden_dim * num_heads, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x_dict, edge_dict):
        """
        x_dict: { node_type: FloatTensor [num_nodes, in_dim] }
        edge_dict: { (src_type, e_type, dst_type): [(src, dst), ...], ... }
        """
        # 1) Project each node type to hidden_dim
        data_dict = {
            ntype: self.input_proj(x) for ntype, x in x_dict.items()
        }

        # 2) For each hidden layer, build HeteroConv if not already, then apply it
        for layer_idx in range(self.num_layers):
            if self.hetero_conv_layers[layer_idx] is None:
                # Create a GATConv for each edge_type
                conv_dict = {}
                for edge_type in edge_dict.keys():
                    conv_dict[edge_type] = GATConv(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        heads=self.num_heads,
                        concat=True,       # => hidden_dim * num_heads
                        negative_slope=0.2,
                        dropout=0.0,
                        add_self_loops=False
                    )
                # Wrap them in HeteroConv (with 'sum' aggregation)
                self.hetero_conv_layers[layer_idx] = HeteroConv(conv_dict, aggr='sum').to(self.device)

            # Apply HeteroConv
            data_dict = self.hetero_conv_layers[layer_idx](
                x_dict=data_dict,
                edge_index_dict=edge_dict
            )

            # Apply non-linearity (ELU) to each node type
            for ntype in data_dict:
                data_dict[ntype] = F.elu(data_dict[ntype])

        # 3) Output layer for 'user' node classification
        if 'user' in data_dict:
            data_dict['user'] = self.out_layer(data_dict['user'])

        return data_dict

###############################################################################
# 8) METRICS
###############################################################################
def compute_metrics(logits, labels_t, loss_fn):
    """
    logits : Tensor [N, 2] for binary classification
    labels_t: Tensor [N] with {0,1}
    Returns { 'loss', 'acc', 'auc' }
    """
    loss_val = loss_fn(logits, labels_t).item()
    preds = torch.argmax(logits, dim=1)
    acc_val = accuracy_score(labels_t.cpu().numpy(), preds.cpu().numpy())
    # For AUC, we need the probability of the positive class (index=1)
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    # If the test set has only one class, roc_auc_score can fail => fallback
    labels_np = labels_t.cpu().numpy()
    if len(set(labels_np)) > 1:
        auc_val = roc_auc_score(labels_np, probs)
    else:
        auc_val = 0.5
    return {'loss': loss_val, 'acc': acc_val, 'auc': auc_val}

###############################################################################
# 9) TRAINING LOOP (CHUNKED)
###############################################################################
def train_in_chunks(
    node_info_path,
    node_labels_path,
    embed_path,
    edges_path,
    embedding_dim=32,
    chunk_size=100000,
    num_epochs=2,
    seed=42,
    test_size_each=100,
    test4_ratio=0.4
):
    """
    Main function that:
      1. Loads node types, labels, embeddings, edges
      2. Splits user nodes into test1, test2, test3, test4, and train
      3. Prints stats
      4. Builds multi-layer GAT model
      5. For each epoch:
         - Chunks edges => build subgraph => forward => train (if in train set)
         - Compute metrics for each test set => accumulate => log
      6. Saves checkpoint
    """
    # 1) Seeds
    set_random_seeds(seed)

    # 2) Device & Logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device: {device}")
    print(f"Running on device: {device}")

    # 3) Load all CSV data
    node_type_map = build_node_type_map(node_info_path)
    labels_map = load_labels(node_labels_path)
    all_embeddings = load_all_embeddings(embed_path, embedding_dim)
    edges_list = load_all_edges(edges_path)

    # 4) Compute user degrees => split => test1, test2, test3, test4, train
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
        save_dir      = "."
    )

    # Print stats
    print_dataset_stats("train", train_set, deg_map, user_deg_map, tweet_deg_map)
    print_dataset_stats("test1", test1, deg_map, user_deg_map, tweet_deg_map)
    print_dataset_stats("test2", test2, deg_map, user_deg_map, tweet_deg_map)
    print_dataset_stats("test3", test3, deg_map, user_deg_map, tweet_deg_map)
    print_dataset_stats("test4", test4, deg_map, user_deg_map, tweet_deg_map)
    all_test = test1.union(test2).union(test3).union(test4)
    print_dataset_stats("all_test", all_test, deg_map, user_deg_map, tweet_deg_map)

    # 5) Create the multi-layer GAT model
    num_classes = 2  # sybil=1, benign=0
    model = MultiLayerHeteroGAT(
        in_dim=embedding_dim,
        hidden_dim=64,
        out_dim=num_classes,
        num_heads=4,    # # of attention heads
        num_layers=4    # 4 hidden layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # 6) Training epochs
    for epoch in range(num_epochs):
        logging.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        epoch_train_loss = 0.0

        # We'll track metrics for each test set across batches
        test_metrics_dict = {
            'test1': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'count': 0},
            'test2': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'count': 0},
            'test3': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'count': 0},
            'test4': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'count': 0},
        }

        # Process edges in chunks
        edge_chunks = chunk_edges_in_memory(edges_list, chunk_size=chunk_size)
        for batch_i, chunk_edges_ in enumerate(edge_chunks, start=1):
            # Build subgraph
            edge_dict, unique_nodes = build_subgraph(chunk_edges_, node_type_map)

            # Build x_dict
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

                arr_size = (len(g_list), embedding_dim)
                big_np = np.zeros(arr_size, dtype=np.float32)
                for i, g in enumerate(g_list):
                    if g in all_embeddings:
                        big_np[i] = all_embeddings[g]
                x_dict[ntype] = torch.from_numpy(big_np).to(device)

            # Convert edges to local indices
            local_edge_dict = defaultdict(list)
            for (src_t, e_type, dst_t), e_list in edge_dict.items():
                s_map = local_id_map[src_t]
                d_map = local_id_map[dst_t]
                for (src, dst) in e_list:
                    if src in s_map and dst in d_map:
                        local_edge_dict[(src_t, e_type, dst_t)].append((s_map[src], d_map[dst]))

            # Forward pass
            out_dict = model(x_dict, local_edge_dict)

            # Train on user nodes that are in train_set
            loss = torch.tensor(0.0, device=device)
            num_labeled_train = 0
            if 'user' in node_lists:
                user_out = out_dict['user']
                user_map = local_id_map['user']

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

                if train_indices:
                    train_idx_t = torch.tensor(train_indices, dtype=torch.long, device=device)
                    train_lbl_t = torch.tensor(train_labels, dtype=torch.long, device=device)
                    logits_train = user_out[train_idx_t]
                    loss = loss_fn(logits_train, train_lbl_t)
                    num_labeled_train = len(train_indices)

            # Backprop
            if num_labeled_train > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item()

            # Evaluate on test sets
            if 'user' in node_lists:
                user_out = out_dict['user']
                user_map = local_id_map['user']

                def evaluate_testset(test_nodes, set_key):
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
                        test_metrics_dict[set_key]['loss'] += m['loss']
                        test_metrics_dict[set_key]['acc']  += m['acc']
                        test_metrics_dict[set_key]['auc']  += m['auc']
                        test_metrics_dict[set_key]['count'] += 1

                # Evaluate each test set
                evaluate_testset(test1, 'test1')
                evaluate_testset(test2, 'test2')
                evaluate_testset(test3, 'test3')
                evaluate_testset(test4, 'test4')

            msg = (f"  Batch {batch_i}: edges={len(chunk_edges_)}, "
                   f"train_users={num_labeled_train}, loss={loss.item():.4f}")
            logging.info(msg)
            print(msg)

        # End of epoch => average metrics
        avg_train_loss = epoch_train_loss / max(batch_i, 1)

        # For each test set, average across all batches that had test samples
        results_str = []
        for k in ['test1', 'test2', 'test3', 'test4']:
            c = test_metrics_dict[k]['count']
            if c > 0:
                test_metrics_dict[k]['loss'] /= c
                test_metrics_dict[k]['acc']  /= c
                test_metrics_dict[k]['auc']  /= c
            results_str.append(
                f"{k}: loss={test_metrics_dict[k]['loss']:.4f}, "
                f"acc={test_metrics_dict[k]['acc']:.4f}, "
                f"auc={test_metrics_dict[k]['auc']:.4f}"
            )

        msg_epoch = (f"[Epoch {epoch+1}] AvgTrainLoss={avg_train_loss:.4f} | "
                     + " | ".join(results_str))
        logging.info(msg_epoch)
        print(msg_epoch)

        # Optionally save model checkpoint each epoch
        torch.save(model.state_dict(), "training_time_hetero_gnn_model.pth")

    return model

###############################################################################
# 10) MAIN
###############################################################################
if __name__ == "__main__":
    # Example paths: adjust these based on your environment
    node_info_path   = "node_information.csv"
    node_labels_path = "node_labels.csv"
    embed_path       = "node_embeddings.csv"
    edges_path       = "edges.csv"

    # Hyperparameters
    chunk_size = 100000
    num_epochs = 20
    embedding_dim = 32  # must match your embedding file dimension

    trained_model = train_in_chunks(
        node_info_path=node_info_path,
        node_labels_path=node_labels_path,
        embed_path=embed_path,
        edges_path=edges_path,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        num_epochs=num_epochs,
        seed=42,
        test_size_each=100,   # test1/test2/test3 => 100 each
        test4_ratio=0.4       # 40% of remaining go into test4
    )

    # Save final model checkpoint with metadata
    checkpoint = {
        'state_dict': trained_model.state_dict(),
        'metadata': {
            'trained_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': '4-layer Hetero-GAT model with extended metadata.',
            'num_epochs': num_epochs,
            'chunk_size': chunk_size
        }
    }
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"hetero_gnn_model_{timestamp}.pth"
    torch.save(checkpoint, filename)
    logging.info(f"Final model saved to {filename}")
    print(f"Final model saved to {filename}")
