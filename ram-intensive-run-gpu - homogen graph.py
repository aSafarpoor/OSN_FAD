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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


# PyG imports
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

###############################################################################
# 1) SETUP LOGGING
###############################################################################
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'my_log_file_{timestamp}.log', mode='a', encoding='utf-8'),
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
def load_labels(node_labels_path):
    """
    Reads node_labels.csv (columns: node_id, label)
    Returns { node_id -> label_str }
    """
    labels_map = {}
    with open(node_labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row['node_id'])
            label_str = row['label']
            labels_map[nid] = label_str
    return labels_map

def load_all_embeddings(embed_path, embedding_dim):
    """
    Reads embeddings from the new file (e.g. ... .csv).
    Assumes columns: node_id, embedding_str
    where embedding_str is "val1,val2,..."
    Returns { node_id -> np.array([...], dtype=float32) }
    """
    all_embeddings = {}
    with open(embed_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if any
        for row_vals in reader:
            if len(row_vals) < 2:
                continue
            row_node_id = int(row_vals[0])
            embedding_str = row_vals[1]
            emb_list = [float(x.strip()) for x in embedding_str.split(",")[:embedding_dim]]
            emb_array = np.array(emb_list, dtype=np.float32)
            all_embeddings[row_node_id] = emb_array
    return all_embeddings

def load_user_user_edges(edges_path, valid_user_ids):
    """
    Reads edges.csv (columns: source, target, attributes)
    - Only keep edges where both source and target are in valid_user_ids.
    Returns a list of tuples (u1, u2).
    """
    edges_list = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row['source'])
            dst = int(row['target'])
            if src in valid_user_ids and dst in valid_user_ids:
                edges_list.append((src, dst))
    print("User-User edges read:", len(edges_list))
    return edges_list

###############################################################################
# 4) COMPUTE USER DEGREES & SPLIT LOGIC
###############################################################################
def compute_user_degrees_for_benign_sybil_users(edges_list, labels_map):
    """
    We only have user nodes and user-user edges, so:
      deg_map[u] = number of neighbors
      user_deg_map[u] = same as deg_map[u]
      tweet_deg_map[u] = 0 (no tweets left, but we preserve shape of code)
    Returns:
      user_ids: list of user nodes that are labeled benign/sybil
      deg_map, user_deg_map, tweet_deg_map
    """
    valid_user_nodes = set()
    for nid, lab in labels_map.items():
        if lab in ['benign', 'sybil']:
            valid_user_nodes.add(nid)

    deg_map = defaultdict(int)
    user_deg_map = defaultdict(int)
    tweet_deg_map = defaultdict(int)  # always zero

    for (src, dst) in edges_list:
        # increment deg if user is in valid set
        if src in valid_user_nodes:
            deg_map[src] += 1
            user_deg_map[src] += 1
        if dst in valid_user_nodes:
            deg_map[dst] += 1
            user_deg_map[dst] += 1

    # fill missing zeros for users not in edges
    for u in valid_user_nodes:
        deg_map.setdefault(u, 0)
        user_deg_map.setdefault(u, 0)
        tweet_deg_map[u] = 0  # always 0

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
    Splits the user_ids into 4 sets:
      test1: 100 with least user_deg
      test2: 100 with least tweet_deg (which is 0 => effectively random if there's a tie)
      test3: 100 with least (user_deg + tweet_deg) => basically same as user_deg
      test4: 40% of the REMAIN + test1 + test2 + test3
      train: everything else
    If files exist, read from disk. Otherwise create them.
    """
    f_test1 = os.path.join(save_dir, 'testset1.txt')
    f_test2 = os.path.join(save_dir, 'testset2.txt')
    f_test3 = os.path.join(save_dir, 'testset3.txt')
    f_test4 = os.path.join(save_dir, 'testset4.txt')
    f_train = os.path.join(save_dir, 'trainset.txt')

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

    # filter only benign/sybil
    candidate_nodes = []
    for uid in user_ids:
        lab = labels_map.get(uid, None)
        if lab in ['benign', 'sybil']:
            candidate_nodes.append(uid)

    # test1: 100 least user_deg
    test1_sorted = sorted(candidate_nodes, key=lambda x: user_deg_map.get(x, 0))
    test1 = set(test1_sorted[:test_size_each])

    # test2: 100 least tweet_deg (always 0 => effectively first 100)
    test2_sorted = sorted(candidate_nodes, key=lambda x: tweet_deg_map.get(x, 0))
    test2 = set(test2_sorted[:test_size_each])

    # test3: 100 least (user_deg + tweet_deg) => same as user_deg
    test3_sorted = sorted(candidate_nodes, key=lambda x: (user_deg_map[x] + tweet_deg_map[x]))
    test3 = set(test3_sorted[:test_size_each])

    combined_123 = test1.union(test2).union(test3)

    remain = [u for u in candidate_nodes if u not in combined_123]
    random.shuffle(remain)
    test4_size = int(len(remain) * test4_ratio)
    test4_part = remain[:test4_size]
    test4 = set(test4_part).union(combined_123)

    train = set(candidate_nodes) - test4

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
    if not node_ids:
        print(f"[{name}] is empty.")
        return
    deg_vals = [deg_map[n] for n in node_ids]
    user_vals = [user_deg_map[n] for n in node_ids]
    tweet_vals = [tweet_deg_map[n] for n in node_ids]  # always 0
    print(
        f"[{name}] count={len(node_ids):5d}, "
        f"avgDeg={np.mean(deg_vals):.2f}, "
        f"avgUserDeg={np.mean(user_vals):.2f}, "
        f"avgTweetDeg={np.mean(tweet_vals):.2f}"
    )

###############################################################################
# 6) CHUNKER
###############################################################################
def chunk_edges_in_memory(edges_list, chunk_size=100000):
    """
    Yields sublists (chunks) of edges_list, each up to `chunk_size`.
    """
    for i in range(0, len(edges_list), chunk_size):
        yield edges_list[i:i + chunk_size]

###############################################################################
# 7) MULTI-LAYER HOMOGENEOUS GAT
###############################################################################
class MultiLayerGAT(nn.Module):
    """
    A multi-layer GAT for a homogeneous user-user graph.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, num_layers=4):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[MultiLayerGAT] Using device: {self.device}')

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # First projection layer
        self.proj = nn.Linear(in_dim, hidden_dim)

        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in_dim = hidden_dim * num_heads if layer_idx > 0 else hidden_dim
            conv = GATConv(
                in_channels=layer_in_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,
                negative_slope=0.2,
                dropout=0.0,
                add_self_loops=False
            )
            self.gat_layers.append(conv)

        # Final classification layer
        self.out_layer = nn.Linear(hidden_dim * num_heads, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x, edge_index):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        """
        # Project
        x = self.proj(x)  # [N, hidden_dim]

        # Pass through GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)  # [N, hidden_dim * num_heads]
            x = F.elu(x)

        # Final classification
        x = self.out_layer(x)  # [N, out_dim]
        return x

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
# 9) TRAINING LOOP (CHUNKED)
###############################################################################
def train_in_chunks(
    node_labels_path,
    embed_path,
    edges_path,
    embedding_dim=32,
    chunk_size=10000,
    num_epochs=2,
    seed=42,
    test_size_each=100,
    test4_ratio=0.4
):
    """
    Main function that:
      1. Loads labels, embeddings, user-user edges
      2. Splits user nodes into test1, test2, test3, test4, and train
      3. Builds multi-layer GAT model
      4. For each epoch, chunk edges => subgraph => forward => train
      5. Evaluate on test sets each chunk => average
    """
    set_random_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device: {device}")
    print(f"Running on device: {device}")

    # 1) Load labels & embeddings
    labels_map = load_labels(node_labels_path)
    all_embeddings = load_all_embeddings(embed_path, embedding_dim)

    # Create set of all user IDs from labels that are benign/sybil
    # (If your labels contain more classes, adapt as needed)
    valid_user_ids = {nid for nid, lab in labels_map.items() if lab in ['benign','sybil']}

    # 2) Load user-user edges
    edges_list = load_user_user_edges(edges_path, valid_user_ids)

    # Compute degrees => split => test sets
    user_ids, deg_map, user_deg_map, tweet_deg_map = compute_user_degrees_for_benign_sybil_users(
        edges_list, labels_map
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

    # 3) Create homogeneous GAT model
    num_classes = 2  # sybil=1, benign=0
    model = MultiLayerGAT(
        in_dim=embedding_dim,
        hidden_dim=64,
        out_dim=num_classes,
        num_heads=4,
        num_layers=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # 4) Training
    for epoch in range(num_epochs):
        logging.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        epoch_train_loss = 0.0

        # We'll track metrics for each test set across chunks
        test_metrics_dict = {
            'test1': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'count': 0},
            'test2': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'count': 0},
            'test3': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'count': 0},
            'test4': {'loss': 0.0, 'acc': 0.0, 'auc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'count': 0},
        }

        edge_chunks = chunk_edges_in_memory(edges_list, chunk_size=chunk_size)
        for batch_i, chunk_edges_ in enumerate(edge_chunks, start=1):
            # Build subgraph (only for the edges in this chunk)
            # 1) unique nodes in this chunk
            unique_nodes = set()
            for (u1, u2) in chunk_edges_:
                unique_nodes.add(u1)
                unique_nodes.add(u2)

            # 2) build a local index mapping
            sorted_nodes = sorted(unique_nodes)
            local_id_map = {g: i for i, g in enumerate(sorted_nodes)}

            # 3) build x
            x_np = np.zeros((len(sorted_nodes), embedding_dim), dtype=np.float32)
            for i, g in enumerate(sorted_nodes):
                if g in all_embeddings:
                    x_np[i] = all_embeddings[g]
            x_t = torch.from_numpy(x_np).to(device)

            # 4) build edge_index
            edge_src = []
            edge_dst = []
            for (u1, u2) in chunk_edges_:
                edge_src.append(local_id_map[u1])
                edge_dst.append(local_id_map[u2])
            # make it bidirectional (if your data isn't directed)
            # If your edges are truly directed, remove the duplication
            edge_src.extend(edge_dst)
            edge_dst.extend(edge_src[:len(edge_dst)])

            edge_index_t = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)

            # 5) forward pass
            out_logits = model(x_t, edge_index_t)

            # 6) compute train loss (only on nodes in train_set with labels)
            loss = torch.tensor(0.0, device=device)
            train_indices = []
            train_labels = []
            for g in sorted_nodes:
                if g in train_set and g in labels_map:
                    lab = labels_map[g]
                    if lab == 'sybil':
                        train_indices.append(local_id_map[g])
                        train_labels.append(1)
                    elif lab == 'benign':
                        train_indices.append(local_id_map[g])
                        train_labels.append(0)

            if train_indices:
                train_idx_t = torch.tensor(train_indices, dtype=torch.long, device=device)
                train_lbl_t = torch.tensor(train_labels, dtype=torch.long, device=device)
                logits_train = out_logits[train_idx_t]
                loss = loss_fn(logits_train, train_lbl_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item()

            # 7) evaluate on each test set
            def evaluate_testset(test_nodes, key):
                idxs = []
                lbls = []
                for g in sorted_nodes:
                    if g in test_nodes and g in labels_map:
                        lab = labels_map[g]
                        if lab == 'sybil':
                            idxs.append(local_id_map[g])
                            lbls.append(1)
                        elif lab == 'benign':
                            idxs.append(local_id_map[g])
                            lbls.append(0)
                if idxs:
                    idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
                    lbls_t = torch.tensor(lbls, dtype=torch.long, device=device)
                    logits_test = out_logits[idxs_t]
                    m = compute_metrics(logits_test, lbls_t, loss_fn)
                    test_metrics_dict[key]['loss'] += m['loss']
                    test_metrics_dict[key]['acc']  += m['acc']
                    test_metrics_dict[key]['auc']  += m['auc']
                    test_metrics_dict[key]['prec'] += m['prec']
                    test_metrics_dict[key]['rec']  += m['rec']
                    test_metrics_dict[key]['f1']   += m['f1']
                    test_metrics_dict[key]['count'] += 1

            evaluate_testset(test1, 'test1')
            evaluate_testset(test2, 'test2')
            evaluate_testset(test3, 'test3')
            evaluate_testset(test4, 'test4')

            msg = (f"  Batch {batch_i}: edges={len(chunk_edges_)}, "
                   f"loss={loss.item():.4f}")
            logging.info(msg)

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

        # save checkpoint
        torch.save(model.state_dict(), "training_time_gnn_model.pth")

    return model

###############################################################################
# 10) MAIN
###############################################################################
if __name__ == "__main__":
    node_labels_path = "node_labels.csv"
    embed_path       = "new_embeddings_none_kg.csv"   # <--- CHANGED HERE
    edges_path       = "sorted_edges.csv"

    chunk_size = 100000
    num_epochs = 100
    embedding_dim = 32  # must match your embedding dimension

    trained_model = train_in_chunks(
        node_labels_path=node_labels_path,
        embed_path=embed_path,
        edges_path=edges_path,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        num_epochs=num_epochs,
        seed=42,
        test_size_each=100,
        test4_ratio=0.4
    )

    # Save final model checkpoint with metadata
    checkpoint = {
        'state_dict': trained_model.state_dict(),
        'metadata': {
            'trained_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Multi-layer GAT on user-user homogeneous graph.',
            'num_epochs': num_epochs,
            'chunk_size': chunk_size
        }
    }
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"gnn_model_{timestamp}.pth"
    torch.save(checkpoint, filename)
    logging.info(f"Final model saved to {filename}")
    print(f"Final model saved to {filename}")