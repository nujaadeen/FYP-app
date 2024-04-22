import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool 

class ParallelGAT(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            mlp_dims:list[int],
            skip_connections: bool,
            heads: int=1,
            dropout: float=0.3
    ):
        super(ParallelGAT, self).__init__()

        self.num_layers = num_layers
        self.heads = heads
        self.skip_connections = skip_connections
        self.dropout = dropout
        self.act_fn = nn.GELU()

        # Graph Attention Blocks
        self.ab_att_blocks = nn.ModuleList()
        self.ab_att_blocks.append(GATConv(in_channels, hidden_channels, heads=1, dropout=dropout))
        for _ in range(num_layers -  1):
            self.ab_att_blocks.append(GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout))

        self.ab_bn = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

        self.ag_att_blocks = nn.ModuleList()
        self.ag_att_blocks.append(GATConv(in_channels, hidden_channels, heads=1, dropout=dropout))
        for _ in range(num_layers -  1):
            self.ag_att_blocks.append(GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout))

        self.ag_bn = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

        # Final Regressor
        self.mlp = self.create_mlp([hidden_channels * 2] + mlp_dims)
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
    
    def forward(self, x_ab, edge_index_ab, batch_ab, x_ag, edge_index_ag, batch_ag):
        # GAT Layers for Antibody
        x_ab_initial = x_ab
        x_ag_initial = x_ag
        for i in range(self.num_layers):
            if (i != 0) and (self.skip_connections):
                x_ab += x_ab_initial
            x_ab = self.ab_att_blocks[i](x_ab, edge_index_ab) 
            x_ab = self.act_fn(x_ab)
            x_ab = self.ab_bn[i](x_ab)

        # GAT Layers for Antigen
        for i in range(self.num_layers):
            if (i != 0) and (self.skip_connections):
                x_ag += x_ag_initial
            x_ag = self.ag_att_blocks[i](x_ag, edge_index_ag)
            x_ag = self.act_fn(x_ag)
            x_ag = self.ag_bn[i](x_ag)

        # Readout Layer
        x_ab = global_mean_pool(x_ab, batch_ab)
        x_ag = global_mean_pool(x_ag, batch_ag)

        # Final Regressor
        x = torch.cat((x_ab, x_ag), dim=1)
        x = self.mlp(x)
        x_reg = self.out_reg(x)
        x_cls = torch.sigmoid(self.out_cls(x))

        return x_reg, x_cls