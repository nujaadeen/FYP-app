# Torch
import torch
from torch import nn

# Torch Geometric
from torch_geometric.nn import GATConv, global_mean_pool 

class ComplexGAT(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            mlp_dims: list[int],
            skip_connections: bool, 
            separate_pooling: bool,
            heads: int=1,
            dropout: float=0.3
    ):
        super(ComplexGAT, self).__init__()

        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.separate_pooling = separate_pooling
        self.heads = heads
        self.dropout = dropout
        self.act_fn = nn.Tanh()

        # Graph Attention Blocks
        self.att_blocks = nn.ModuleList()
        self.att_blocks.append(GATConv(in_channels, hidden_channels, heads=1, dropout=dropout))
        for _ in range(num_layers - 1):
            self.att_blocks.append(GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout))

        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

        # Final Regressor
        if separate_pooling:
            self.mlp = self.create_mlp([hidden_channels * 2] + mlp_dims)
        else:
            self.mlp = self.create_mlp([hidden_channels] + mlp_dims)
        self.out_reg = nn.Linear(mlp_dims[-1], 1)
        self.out_cls = nn.Linear(mlp_dims[-1], 1)

    def create_layer(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_dim, out_dim),
            self.act_fn,
            nn.BatchNorm1d(out_dim),
        )

    def create_mlp(self, dims: list[int]):
        return nn.Sequential(
            *[self.create_layer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
    
    def forward(self, x, x_type, edge_index, batch):        
        # GAT Layers 
        x_initial = x
        for i in range(self.num_layers):
            if (i != 0) and (self.skip_connections):
                x += x_initial
            x = self.att_blocks[i](x, edge_index)
            x = self.act_fn(x)
            x = self.bn[i](x)

        # Readout Layer
        if self.separate_pooling:
            x_ag = x[torch.where(x_type == 0)[0]]  # Extract features where x_type is zero
            x_ab = x[torch.where(x_type == 1)[0]]   # Extract features where x_type is one
        
            # Compute batch values for x_zeros and x_ones
            batch_ag = batch[torch.where(x_type == 0)[0]]
            batch_ab = batch[torch.where(x_type == 1)[0]]

            x_ab = global_mean_pool(x_ab, batch_ab)
            x_ag = global_mean_pool(x_ag, batch_ag)

            x = torch.cat((x_ab, x_ag), dim=1)
        else:
            x = global_mean_pool(x, batch)
        
        # Final Regressor
        x = self.mlp(x)
        
        x_reg = self.out_reg(x)
        x_cls = torch.sigmoid(self.out_cls(x))

        return x_reg, x_cls