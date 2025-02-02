import torch
from torch import nn
from torch_geometric.nn import RGCNConv,GCNConv,GATConv
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from layer import SimpleHGNLayer, RGTLayer
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


class SheafGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_edge_types=2, dropout=0.6, negative_slope=0.2, bias=True):
        """
        A sheaf-based graph attention layer that uses separate parameters per edge type.
        
        For each edge (i, j), the layer:
          1. Transforms node features with a weight matrix (or the same one if there is only one edge type).
          2. Computes an attention coefficient via a learnable attention vector.
          3. Aggregates messages using these coefficients.
          4. Adds a self-loop contribution.
        
        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            num_edge_types (int): Number of distinct edge types.
            dropout (float): Dropout probability on attention coefficients.
            negative_slope (float): Negative slope for the LeakyReLU in the attention mechanism.
            bias (bool): Whether to add a bias in the self-loop transformation.
        """
        super(SheafGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        self.negative_slope = negative_slope

        # One weight matrix per edge type (or one if num_edge_types==1)
        self.weight = nn.Parameter(torch.Tensor(num_edge_types, in_channels, out_channels))
        # One attention vector per edge type; if only one edge type, this will be used for all edges.
        self.att = nn.Parameter(torch.Tensor(num_edge_types, 2 * out_channels))
        # Self-loop (root) transformation.
        self.root = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        self.root.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        """
        Forward pass.
        
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (LongTensor): Edge indices of shape [2, num_edges] (src, dst).
            edge_type (LongTensor): Tensor of shape [num_edges] with values in [0, num_edge_types-1].
        
        Returns:
            Tensor: Updated node features [num_nodes, out_channels].
        """
        num_nodes = x.size(0)
        src, dst = edge_index  # each of shape [num_edges]

        if self.num_edge_types == 1:
            # Single edge type: use the first (and only) weight and attention parameters.
            W = self.weight[0]              # [in_channels, out_channels]
            h_src = torch.matmul(x[src], W)  # [num_edges, out_channels]
            h_dst = torch.matmul(x[dst], W)  # [num_edges, out_channels]
            a = self.att[0]                 # [2*out_channels]
            # Broadcast the attention vector to each edge.
            cat = torch.cat([h_src, h_dst], dim=1)  # [num_edges, 2*out_channels]
            e = F.leaky_relu((cat * a).sum(dim=1), negative_slope=self.negative_slope)
        else:
            # Multiple edge types: use batched operations.
            # Ensure the weight tensor is contiguous.
            W = self.weight[edge_type].contiguous()  # [num_edges, in_channels, out_channels]
            # Compute transformed features via batched matrix multiplication.
            h_src = torch.bmm(x[src].unsqueeze(1), W).squeeze(1)  # [num_edges, out_channels]
            h_dst = torch.bmm(x[dst].unsqueeze(1), W).squeeze(1)    # [num_edges, out_channels]
            a = self.att[edge_type]                                # [num_edges, 2*out_channels]
            cat = torch.cat([h_src, h_dst], dim=1)                  # [num_edges, 2*out_channels]
            e = F.leaky_relu((cat * a).sum(dim=1), negative_slope=self.negative_slope)

        # Normalize the attention coefficients over incoming edges for each target node.
        alpha = softmax(e, dst, num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Multiply h_src by the normalized attention coefficients.
        h_src = h_src * alpha.unsqueeze(-1)

        # Create an aggregation tensor with the same dtype as x.
        out = x.new_zeros(num_nodes, self.out_channels)
        # Ensure that h_src is in the same dtype as out.
        out = out.index_add(0, dst, h_src.to(out.dtype))
        # Add the self-loop contribution.
        out = out + self.root(x)
        return out

########################################################################
# SheafGAT Module Using the SheafGATConv Layers
########################################################################

class SheafGAT(pl.LightningModule):
    def __init__(self, args):
        """
        PyTorch Lightning module for a sheaf-based graph attention network.
        
        Architecture:
          - Input linear layer with LeakyReLU.
          - Two sheaf-based graph attention layers.
          - An intermediate linear layer with LeakyReLU.
          - A final classification layer.
        
        Args:
            args: An object (e.g., from argparse) with attributes:
                  - lr: Learning rate.
                  - l2_reg: Weight decay.
                  - test_batch_size: Batch size for testing.
                  - features_num: Number of input feature dimensions.
                  - linear_channels: Hidden size after the input layer.
                  - out_channel: Output channels from the sheafGAT layers.
                  - out_dim: Number of classes.
                  - dropout: Dropout probability.
                  - num_edge_types: Number of distinct edge types.
        """
        super(SheafGAT, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num

        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        # Input linear layer.
        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )

        # Two sheaf-based graph attention layers.
        self.sheafgat1 = SheafGATConv(
            in_channels=args.linear_channels,
            out_channels=args.linear_channels,
            num_edge_types=2,#args.num_edge_types,
            dropout=args.dropout
        )
        self.sheafgat2 = SheafGATConv(
            in_channels=args.linear_channels,
            out_channels=args.out_channel,
            num_edge_types=2,#args.num_edge_types,
            dropout=args.dropout
        )

        # Further processing and classification.
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.out_channel, 64),
            nn.LeakyReLU()
        )
        self.out2 = nn.Linear(64, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()

        self.init_weight()

    def init_weight(self):
        # Initialize Linear layers with Kaiming Uniform.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y
        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        # Forward pass.
        user_features = self.linear_relu_input(user_features)
        user_features = self.drop(self.sheafgat1(user_features, edge_index, edge_type))
        user_features = self.sheafgat2(user_features, edge_index, edge_type)
        user_features = self.drop(self.linear_relu_output1(user_features))
        pred = self.out2(user_features)

        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y
            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.sheafgat1(user_features, edge_index, edge_type))
            user_features = self.sheafgat2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.sheafgat1(user_features, edge_index, edge_type))
            user_features = self.sheafgat2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test, pred_test)
            f1 = f1_score(label_test, pred_test, average='macro')
            precision = precision_score(label_test, pred_test, average='macro')
            recall = recall_score(label_test, pred_test, average='macro')

            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}




class SHGN(pl.LightningModule):
    def __init__(self, args):
        super(SHGN, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear1 = nn.Linear(args.features_num, args.linear_channels)
        self.HGN_layer1 = SimpleHGNLayer(num_edge_type=args.relation_num,
                                         in_channels=args.linear_channels,
                                         out_channels=args.linear_channels,
                                         rel_dim=args.rel_dim,
                                         beta=args.beta)
        self.HGN_layer2 = SimpleHGNLayer(num_edge_type=args.relation_num,
                                         in_channels=args.linear_channels,
                                         out_channels=args.out_channel,
                                         rel_dim=args.rel_dim,
                                         beta=args.beta,
                                         final_layer=True)

        self.out1 = torch.nn.Linear(args.out_channel, 64)
        self.out2 = torch.nn.Linear(64, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.drop(self.ReLU(self.linear1(user_features)))
        user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
        user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
            user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.drop(self.ReLU(self.linear1(user_features)))
            user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
            user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')

            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }






class BotRGCN(pl.LightningModule):
    def __init__(self, args):
        super(BotRGCN, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(args.linear_channels, args.linear_channels, num_relations=args.relation_num)
        self.rgcn2 = RGCNConv(args.linear_channels, args.out_channel, num_relations=args.relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.out_channel, 64),
            nn.LeakyReLU()
        )

        self.out2 = torch.nn.Linear(64, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.linear_relu_input(user_features)
        user_features = self.drop(self.rgcn1(user_features, edge_index, edge_type))
        user_features = self.rgcn2(user_features, edge_index, edge_type)
        user_features = self.linear_relu_output1(user_features)
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.rgcn1(user_features, edge_index, edge_type))
            user_features = self.rgcn2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.rgcn1(user_features, edge_index, edge_type))
            user_features = self.rgcn2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')

            print("\nacc: {}".format(acc),
                "f1: {}".format(f1),
                "precision: {}".format(precision),
                "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }



class GAT(pl.LightningModule):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )

        self.gat1 = GATConv(args.linear_channels, int(args.linear_channels / 4), heads=4)
        self.gat2 = GATConv(args.linear_channels, args.out_channel)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.out_channel, 64),
            nn.LeakyReLU()
        )

        self.out2 = torch.nn.Linear(64, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.linear_relu_input(user_features)
        user_features = self.drop(self.gat1(user_features, edge_index))
        user_features = self.gat2(user_features, edge_index)
        user_features = self.drop(self.linear_relu_output1(user_features))
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.gat1(user_features, edge_index))
            user_features = self.gat2(user_features, edge_index)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.gat1(user_features, edge_index))
            user_features = self.gat2(user_features, edge_index)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')

            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }




          
class SheafConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_edge_types=2, bias=True):
        """
        A simple sheaf convolution layer that uses a different weight for each edge type.

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
            num_edge_types (int): Number of distinct edge types.
            bias (bool): Whether to include a bias term in the self-loop transform.
        """
        super(SheafConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        # One weight matrix per edge type: shape [num_edge_types, in_channels, out_channels]
        self.weight = nn.Parameter(torch.Tensor(num_edge_types, in_channels, out_channels))
        # Self-loop transformation (or “root” transform)
        self.root = nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self.root.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        """
        Forward pass of the sheaf convolution.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (LongTensor): Edge index tensor of shape [2, num_edges],
                                     where edge_index[0] holds source node indices and
                                     edge_index[1] holds target node indices.
            edge_type (LongTensor): Edge type tensor of shape [num_edges] with integer values 
                                    in the range [0, num_edge_types - 1].

        Returns:
            Tensor: Updated node features of shape [num_nodes, out_channels].
        """
        num_nodes = x.size(0)
        src, dst = edge_index  # source and target node indices for each edge
        # Gather source node features for all edges: shape [num_edges, in_channels]
        x_src = x[src]

        if self.num_edge_types == 1:
            # When there is only one edge type, use the same weight for all edges.
            weight = self.weight[0]  # shape: [in_channels, out_channels]
            messages = torch.matmul(x_src, weight)  # shape: [num_edges, out_channels]
        else:
            # For each edge, select the weight matrix corresponding to its type.
            # Use contiguous() to ensure the memory layout is suitable for batched matmul.
            W = self.weight[edge_type].contiguous()  # shape: [num_edges, in_channels, out_channels]
            # Compute messages: for each edge, message = x_src @ W.
            messages = torch.bmm(x_src.unsqueeze(1), W).squeeze(1)  # shape: [num_edges, out_channels]

        # Aggregate messages: sum the messages for each target node.
        out = x.new_zeros(num_nodes, self.out_channels)
        out = out.index_add(0, dst, messages)
        # Add self-loop contribution (the node's own transformed features)
        out = out + self.root(x)
        return out

class SheafGCN(pl.LightningModule):
    def __init__(self, args):
        """
        PyTorch Lightning module for a sheaf-based graph convolutional network.
        
        The network applies:
          - A linear transformation with LeakyReLU (input layer)
          - Two sheaf convolution layers (with dropout in between)
          - A final linear classification head
        
        Args:
            args: An object (e.g., from argparse) with attributes:
                  - lr: Learning rate.
                  - l2_reg: L2 regularization (weight decay).
                  - test_batch_size: Batch size for testing.
                  - features_num: Number of node feature dimensions to use.
                  - linear_channels: Hidden channel size after the input layer.
                  - out_channel: Output channel size from the sheaf conv layers.
                  - out_dim: Number of classes (output dimension).
                  - dropout: Dropout probability.
                  - num_edge_types: Number of distinct edge types in your graph.
        """
        super(SheafGCN, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        # Input layer: projects input features to a hidden dimension.
        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )
        # Two sheaf convolution layers.
        self.sheafgcn1 = SheafConv(args.linear_channels, args.linear_channels, num_edge_types=1)
        self.sheafgcn2 = SheafConv(args.linear_channels, args.out_channel, num_edge_types=1)

        # Output layers: further transformation and final classification.
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.out_channel, 64),
            nn.LeakyReLU()
        )
        self.out2 = nn.Linear(64, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        # Initialize all Linear layers using Kaiming Uniform initialization.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        # Extract the necessary inputs.
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y
        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        # Forward pass through the network.
        user_features = self.linear_relu_input(user_features)
        user_features = self.drop(self.sheafgcn1(user_features, edge_index, edge_type))
        user_features = self.sheafgcn2(user_features, edge_index, edge_type)
        user_features = self.drop(self.linear_relu_output1(user_features))
        pred = self.out2(user_features)

        # Use the provided training mask and batch size to select the training nodes.
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y
            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.sheafgcn1(user_features, edge_index, edge_type))
            user_features = self.sheafgcn2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.sheafgcn1(user_features, edge_index, edge_type))
            user_features = self.sheafgcn2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            # For binary classification, assume class 1 probability is of interest.
            self.pred_test_prob.append(pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test, pred_test)
            f1 = f1_score(label_test, pred_test, average='macro')
            precision = precision_score(label_test, pred_test, average='macro')
            recall = recall_score(label_test, pred_test, average='macro')

            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }




class GCN(pl.LightningModule):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )
        self.gcn1 = GCNConv(args.linear_channels, args.linear_channels)
        self.gcn2 = GCNConv(args.linear_channels, args.out_channel)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.out_channel, 64),
            nn.LeakyReLU()
        )

        self.out2 = torch.nn.Linear(64, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.linear_relu_input(user_features)
        user_features = self.drop(self.gcn1(user_features, edge_index))
        user_features = self.gcn2(user_features, edge_index,)
        user_features = self.drop(self.linear_relu_output1(user_features))
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.gcn1(user_features, edge_index))
            user_features = self.gcn2(user_features, edge_index)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.gcn1(user_features, edge_index))
            user_features = self.gcn2(user_features, edge_index)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')


            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }




class RGT(pl.LightningModule):
    def __init__(self, args):
        super(RGT, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )
        self.RGT_layer1 = RGTLayer(num_edge_type=args.relation_num,
                                   in_channel=args.linear_channels,
                                   trans_heads=args.trans_head,
                                   semantic_head=args.semantic_head,
                                   out_channel=args.linear_channels,
                                   dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=args.relation_num,
                                   in_channel=args.linear_channels,
                                   trans_heads=args.trans_head,
                                   semantic_head=args.semantic_head,
                                   out_channel=args.linear_channels,
                                   dropout=args.dropout)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.linear_channels, 64),
            nn.LeakyReLU()
        )

        self.out2 = torch.nn.Linear(64, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.linear_relu_input(user_features)
        user_features = self.RGT_layer1(user_features, edge_index, edge_type)
        user_features = self.RGT_layer2(user_features, edge_index, edge_type)
        user_features = self.drop(self.linear_relu_output1(user_features))
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.RGT_layer1(user_features, edge_index, edge_type)
            user_features = self.RGT_layer2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.RGT_layer1(user_features, edge_index, edge_type)
            user_features = self.RGT_layer2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')


            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }