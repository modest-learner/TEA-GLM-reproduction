import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .layers import SAGEConv


class GraphEncoder(nn.Module):    # first_model, i.e. simple MLP for dimension transformation of OGB/ GIANT node embedding + freezed Llama-7b word embeddings.
    def __init__(self, args, llama_embed):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.GT = GraphSAGE(args)

        self.register_buffer('embed_tokens', llama_embed.detach(), persistent=False)
        self.embed_dim = llama_embed.shape[1]

        self.graph_projector = nn.Sequential(
            nn.Linear(args.gnn_output, self.args.num_token * self.embed_dim),
            )
        # Preserve the released initialization for stage-2 training. Loading
        # the full author checkpoint overwrites these weights at inference.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0., std=1.)
        
    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.GT.parameters()):
            self.GT.eval()
        return self

    def forward(self, input_ids, is_node, graph, return_features=False):
        counts = is_node.sum(dim=1)
        if not t.all(counts == self.args.num_token):
            raise ValueError('Each sample must contain num_token graph placeholders')
        dtype = next(self.GT.parameters()).dtype
        gnn_features = self.GT(graph.x.to(dtype=dtype), graph.edge_index, graph.edge_attr, graph.batch, graph.lp)
        node_embedding = self.graph_projector(gnn_features)
        node_embeddings = node_embedding.view(-1, self.embed_dim) # [bs * 5, dim]

        if node_embedding.shape[0] != input_ids.shape[0]:
            raise ValueError('Graph batch and token batch sizes differ')
        inputs_embeds = F.embedding(input_ids.masked_fill(is_node, 0), self.embed_tokens)
        inputs_embeds[is_node] = node_embeddings.to(inputs_embeds.dtype)

        if return_features:
            # These are the exact states used in this forward pass, before
            # injection into the LLM; no second GNN/projector evaluation.
            return inputs_embeds, {'H_GNN': gnn_features,
                                   'T_graph': node_embedding.view(-1, self.args.num_token, self.embed_dim)}
        return inputs_embeds # [bsz, seq, dim]


class GraphSAGE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_channels = args.gnn_input
        n_layers = args.gt_layers
        hidden_channels = args.att_d_model
        out_channels = args.gnn_output
        edge_dim = args.edge_dim
        num_proj_hidden = out_channels

        gnn_conv = SAGEConv
        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels))
            for i in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels))
            self.convs.append(gnn_conv(hidden_channels, out_channels))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels))

        # non-linear layer for contrastive loss
        self.fc1 = nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, out_channels)

        self.activation = F.relu

        #Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool

    def forward(self, x, edge_index, edge_attr=None, batch=None, lp=None):
        for i, conv in enumerate(self.convs):
            # x = conv(x, edge_index, edge_attr=edge_attr)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)
        
        x = list(unbatch(x, batch))
        xs = []
        assert lp.shape[0] == len(x)
        for i in range(len(x)):
            if lp[i].data.item() == True:
                xs.append((x[i][0] + x[i][1]) / 2)
            else:
                xs.append(x[i][0])
        x = t.stack(xs, dim=0)

        return x

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
