import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import torch.nn.functional as F

# Load data
edge_index = torch.load('edge_index.pt', weights_only=True)
edge_type = torch.load('edge_type.pt', weights_only=True)
features = torch.load('features.pt', weights_only=True)
labels = torch.load('labels_bot.pt', weights_only=True)

# Filter edges by edge_type (0 or 1)
filtered_edges = edge_index[:, (edge_type == 0) | (edge_type == 1)]

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Split data into train/test
human_indices = (labels == 0).nonzero(as_tuple=True)[0]
bot_indices = (labels == 1).nonzero(as_tuple=True)[0]

# Ensure the test set is limited to 500 human and 500 bot nodes
train_human, test_human = train_test_split(human_indices, train_size=20, test_size=1000, random_state=seed)
train_bot, test_bot = train_test_split(bot_indices, train_size=20, test_size=1000, random_state=seed)

# Combine train and test indices
train_indices = torch.cat((train_human, train_bot))
test_indices = torch.cat((test_human, test_bot))


# Define the GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1):
        super(GAT, self).__init__()
        self.layers = torch.nn.ModuleList()
        # Input layer
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        # Output layer
        self.layers.append(GATConv(hidden_channels * heads, out_channels, heads=1))

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x

# Prepare data for PyTorch Geometric
data = Data(x=features, edge_index=filtered_edges, y=labels)

# Create train and test masks
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_indices] = True
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[test_indices] = True

# Initialize model, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(in_channels=features.size(1), hidden_channels=64, out_channels=2, num_layers=3).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item(), out

# Testing loop
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        probs = out[mask][:, 1].cpu().numpy()

        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
        auc = roc_auc_score(y_true, probs)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return acc, auc, f1, precision, recall

# Training the model
x  = 100
for epoch in range(x):
    loss, train_out = train()
    if epoch % 10 == 0 or epoch == x-1:
        train_acc, train_auc, train_f1, train_precision, train_recall = evaluate(data.train_mask)
        test_acc, test_auc, test_f1, test_precision, test_recall = evaluate(data.test_mask)
        print(f"Epoch {epoch}")
        print(f"  Train Loss: {loss:.4f}")
        print(f"  Train Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"  Test Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

print("Training complete.")
