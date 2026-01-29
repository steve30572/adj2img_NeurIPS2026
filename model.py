from typing import Any, Optional

import torch
from torch_geometric.nn import MessagePassing
import torch_geometric

import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from utils import *

class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        cache_shape = (max_seq_len, self.theta.shape[0], 2)
        cache = nn.Parameter(cache, requires_grad=True)
        # cache = torch.cos(cache)
        # self.cache = cache
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.view_as_real(torch.view_as_complex(xshaped) * torch.view_as_complex(rope_cache))

        # # tensor has shape [b, s, n_h, h_d // 2, 2]
        # x_out = torch.stack(
        #     [
        #         xshaped[..., 0] * rope_cache[..., 0]
        #         - xshaped[..., 1] * rope_cache[..., 1],
        #         xshaped[..., 1] * rope_cache[..., 0]
        #         + xshaped[..., 0] * rope_cache[..., 1],
        #     ],
        #     -1,
        # )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class SimpleSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleSAGE, self).__init__()
        self.conv1 = torch_geometric.nn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = torch_geometric.nn.SAGEConv(hidden_dim, output_dim)
    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class SimpleGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
        super(SimpleGAT, self).__init__()
        self.conv1 = torch_geometric.nn.GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = torch_geometric.nn.GATConv(hidden_dim * heads, output_dim, heads=1)
    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
# Define a simple GCN model
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class TanhAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        B, S, E = query.size()
        q = self.q_proj(query).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5  # (B, num_heads, S, S)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # only for heterophily datasets
        # attn_scores = (attn_scores * 2 ) - 1

        
        # attn_scores = 2 * (attn_scores - attn_scores.min(dim=-1, keepdim=True)[0]) / (
        #     attn_scores.max(dim=-1, keepdim=True)[0] - attn_scores.min(dim=-1, keepdim=True)[0] + 1e-8
        # ) - 1  # Scale to (-1, 1)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = attn_scores  # No softmax, just tanh
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, S, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, E)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_probs

# Example: Replace nn.MultiheadAttention in TransformerBlock with TanhAttention
# In TransformerBlock, change:
# self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
# to:
# self.attn = TanhAttention(embed_dim, num_heads, dropout=dropout)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        # self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn = TanhAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        ## Rotary (0825)
        # self.rotary_emb = RotaryPositionalEmbeddings(embed_dim // num_heads, max_seq_len=10)
        # self.num_head = num_heads

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        
        # ## Rotary (0825)
        # attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], self.num_head, -1)
        # attn_out = self.rotary_emb(attn_out)
        # attn_out = attn_out.flatten(2)
        # ##
        x = self.norm1(x + attn_out)
        # x = self.norm1(attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        # x = self.norm2(mlp_out)
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        # self.input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def median_act(self, x, k):
        # x: (batch, seq_len, input_dim)
        # x = self.input_proj(x)

        for i, layer in enumerate(self.layers):
            if i == k:
                return x
            x = layer(x)
        return x


class MeanAggregator(MessagePassing):
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class SimpleAggregator(MessagePassing):
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        # x: [num_nodes, num_features]
        # edge_index: [2, num_edges]
        # Compute degree normalization as in GCN
        row, col = edge_index
        deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        self._norm = norm  # Store for use in message
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        
        return x_j #* self._norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

class SimpleAggregatorNorm(MessagePassing):
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        # x: [num_nodes, num_features]
        # edge_index: [2, num_edges]
        # Compute degree normalization as in GCN
        row, col = edge_index
        deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        self._norm = norm  # Store for use in message
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        
        return x_j * self._norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

class MyModel(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, x, edge_index):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)
        x_input = torch.stack((x, x_2, x_3, x_4), dim=1) # token_representation, 
        x = self.transformer(x_input)
        x = self.output_proj(x)
        return x

    def median_act(self, x, edge_index, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)
        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class MyModel_with_VQ_Norm(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregatorNorm()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.tanh(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z

        img_embed = self.img_embed(img_z)
        x = self.output_proj(img_embed.squeeze(1))
        return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)

        
        # x += self.type_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        # x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))


        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # x_2 = torch.nn.functional.normalize(x_2, p=2, dim=1)
        # x_3 = torch.nn.functional.normalize(x_3, p=2, dim=1)
        
        # x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=1)

        # x += self.hop_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.hop_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x.device)+1)
        # x_3 += self.hop_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x.device)+2)
        # img_embed[:, 0] += self.hop_emb(torch.zeros(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 1] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 2] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device)+1)

        # prefix = self.prefix_token(torch.arange(10, device=x.device))
        # prefix = prefix.unsqueeze(0).expand(x.size(0), -1, -1)

        x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = x_3.unsqueeze(1) # token_representation, 
    

        # x_input = x_2.unsqueeze(1)
        # x_input = torch.cat((x_input, img_embed), dim=1)


        x_input = torch.cat((img_embed, x_input), dim=1)


        # x_input = torch.cat((x_input, img_embed, prefix), dim=1)
        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)
        x = self.output_proj(x)


        # x = self.output_proj(img_embed.squeeze(1))
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class MyModel_with_VQ_Hetero(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregatorNorm()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.tanh(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z

        img_embed = self.img_embed(img_z)
        x = self.output_proj(img_embed.squeeze(1))
        return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        # img = F.dropout(img, p=0.2, training=self.training) ## added
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)

        
        # x_2_agg += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        # x_3_agg += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        


        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # x_2 = torch.nn.functional.normalize(x_2, p=2, dim=1)
        # x_3 = torch.nn.functional.normalize(x_3, p=2, dim=1)
        
        # x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=1)

        # x += self.hop_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.hop_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x.device)+1)
        # x_3 += self.hop_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x.device)+2)
        # img_embed[:, 0] += self.hop_emb(torch.zeros(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 1] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 2] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device)+1)

        # prefix = self.prefix_token(torch.arange(10, device=x.device))
        # prefix = prefix.unsqueeze(0).expand(x.size(0), -1, -1)

        x_input = torch.stack((x, x_2_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = x_3.unsqueeze(1) # token_representation, 
    

        # x_input = x_2.unsqueeze(1)
        # x_input = torch.cat((x_input, img_embed), dim=1)
        x_input = torch.cat((img_embed, x_input), dim=1)
        # x_input = torch.cat((x_input, img_embed, prefix), dim=1)
        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)
        x = self.output_proj(x)


        # x = self.output_proj(img_embed.squeeze(1))
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x


class MyModel_with_VQ(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.tanh(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z

        img_embed = self.img_embed(img_z)
        x = self.output_proj(img_embed.squeeze(1))
        return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)
        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        # x_4 = self.aggregator(x_3, edge_index)

        # x += self.type_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        # x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        # x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # x += self.hop_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.hop_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x.device)+1)
        # x_3 += self.hop_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x.device)+2)
        # img_embed[:, 0] += self.hop_emb(torch.zeros(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 1] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 2] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device)+1)

        x_input = torch.stack((x, x_2, x_3), dim=1) # token_representation, 
        # x_input = x_2.unsqueeze(1)
        x_input = torch.cat((x_input, img_embed), dim=1)

        # x_input = torch.cat((img_embed, x_input), dim=1)

        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)
        x = self.output_proj(x)


        # x = self.output_proj(img_embed.squeeze(1))
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x


class MyModel_with_VQ_concat(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img = nn.Linear(image_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim * 2, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)
        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        # x_4 = self.aggregator(x_3, edge_index)

        x += self.type_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        # x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # x += self.hop_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.hop_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x.device)+1)
        # x_3 += self.hop_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x.device)+2)
        # img_embed[:, 0] += self.hop_emb(torch.zeros(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 1] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 2] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device)+1)

        x_input = torch.stack((x, x_2, x_3), dim=1) # token_representation, 
        x_input = torch.cat((x_input, img_embed), dim=1)
        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)
        x = torch.concat((x[:,2], torch.mean(x[:, 3:], dim=1, keepdim=False)), dim=1)
        x = self.output_proj(x)
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x


class MyModel_without_VQ_Norm(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.tanh(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z

        img_embed = self.img_embed(img_z)
        x = self.output_proj(img_embed.squeeze(1))
        return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)

        
        # x += self.type_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        # x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))


        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # x_2 = torch.nn.functional.normalize(x_2, p=2, dim=1)
        # x_3 = torch.nn.functional.normalize(x_3, p=2, dim=1)
        
        # x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        
        
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=1)

        # x += self.hop_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.hop_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x.device)+1)
        # x_3 += self.hop_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x.device)+2)
        # img_embed[:, 0] += self.hop_emb(torch.zeros(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 1] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 2] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device)+1)

        # prefix = self.prefix_token(torch.arange(10, device=x.device))
        # prefix = prefix.unsqueeze(0).expand(x.size(0), -1, -1)

        x_input = torch.stack((x_2_agg, x_3_agg, x_3), dim=1) # token_representation, 
        # x_input = x_3.unsqueeze(1) # token_representation, 
    

        # x_input = x_2.unsqueeze(1)
        # x_input = torch.cat((x_input, img_embed), dim=1)


        # x_input = torch.cat((img_embed, x_input), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)


        # x_input = torch.cat((x_input, img_embed, prefix), dim=1)
        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)
        x = self.output_proj(x)


        # x = self.output_proj(img_embed.squeeze(1))
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class MyModel_without_VQ_Gate(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # img_y = torch.nn.functional.softmax(img, dim=-1)
        
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z

        # img_embed = self.img_embed(img_z)
        # x = self.output_proj(img_embed.squeeze(1))
        # return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # print(self.img_embed.weight[0,:10])
        if self.training:
            img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        else:
            # img_y = torch.nn.functional.softmax(img, dim=-1)
            img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_embed = torch.matmul(img_y, self.img_embed.weight)
        # img_z = img_y.argmax(dim=-1)
        # print(img_z[0], img_y[0])
        # exit(0)
        # img_embed = self.img_embed(img_z)
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)

        
        # x += self.type_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        # x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))


        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # x_2 = torch.nn.functional.normalize(x_2, p=2, dim=1)
        # x_3 = torch.nn.functional.normalize(x_3, p=2, dim=1)
        
        # x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        
        
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=1)

        # x += self.hop_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.hop_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x.device)+1)
        # x_3 += self.hop_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x.device)+2)
        # img_embed[:, 0] += self.hop_emb(torch.zeros(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 1] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 2] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device)+1)

        # prefix = self.prefix_token(torch.arange(10, device=x.device))
        # prefix = prefix.unsqueeze(0).expand(x.size(0), -1, -1)

        x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = x_3.unsqueeze(1) # token_representation, 
    

        # x_input = x_2.unsqueeze(1)
        # x_input = torch.cat((x_input, img_embed), dim=1)


        # x_input = torch.cat((img_embed, x_input), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)


        # x_input = torch.cat((x_input, img_embed, prefix), dim=1)
        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)

        ### Gating

        x = self.output_proj(x)
        gate = torch.sigmoid(x[:, 0])
        x = x[:, 0] * gate + (1 - gate) * x[:, -1]

        # x = self.output_proj(img_embed.squeeze(1))
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class MyModel_without_VQ_Gate_Mix(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # img_y = torch.nn.functional.softmax(img, dim=-1)

        img_y = img

        
        # img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        # print(img_y[0,0], img_y[200,0])
        # print("*" * 20)
        img_z = img_y.argmax(dim=-1)
        return img_z

        # img_embed = self.img_embed(img_z)
        # x = self.output_proj(img_embed.squeeze(1))
        # return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)

        ### Softmax added
        # img = F.softmax(img, dim=-1)

        img_embed = torch.matmul(img, self.img_embed.weight)
        
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)
        
        
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        

        # x = self.transformer(x_input)
        # return x[:, 0]


        x_input = torch.cat((x_input, img_embed), dim=1)
        # x_input = img_embed
        x = self.transformer(x_input)

        ### Gating

        x = self.output_proj(x)
        gate = torch.sigmoid(x[:, 0])
        x = x[:, 0] * gate + (1 - gate) * x[:, -1]
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class MyModel_without_VQ_Gate_Mix_struct(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # img_y = torch.nn.functional.softmax(img, dim=-1)

        img_y = img

        
        # img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        # print(img_y[0,0], img_y[200,0])
        # print("*" * 20)
        img_z = img_y.argmax(dim=-1)
        return img_z

        # img_embed = self.img_embed(img_z)
        # x = self.output_proj(img_embed.squeeze(1))
        # return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)

        ### Softmax added
        # img = F.softmax(img, dim=-1)

        img_embed = torch.matmul(img, self.img_embed.weight)
        
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)
        
        
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        

        # x = self.transformer(x_input)
        # return x[:, 0]


        # x_input = torch.cat((x_input, img_embed), dim=1)
        # x_input = img_embed
        x = self.transformer(x_input)

        ### Gating

        x = self.output_proj(x)
        # gate = torch.sigmoid(x[:, 0])
        # x = x[:, 0] * gate + (1 - gate) * x[:, -1]
        return x[:, 0]

class MyModel_without_VQ_Gate_Mix_GFM(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # img_y = torch.nn.functional.softmax(img, dim=-1)

        img_y = img

        
        # img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        # print(img_y[0,0], img_y[200,0])
        # print("*" * 20)
        img_z = img_y.argmax(dim=-1)
        return img_z

        # img_embed = self.img_embed(img_z)
        # x = self.output_proj(img_embed.squeeze(1))
        # return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        # img = F.relu(img)
        img = self.input_proj_img2(img)

        ### Softmax added
        # img = F.softmax(img, dim=-1)

        img_embed = torch.matmul(img, self.img_embed.weight)
        
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)
        
        
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x_3, x_3), dim=1) # token_representation, 
        

        # x = self.transformer(x_input)
        # return x[:, 0]


        x_input = torch.cat((x_input, img_embed), dim=1)
        # x_input = img_embed
        x = self.transformer(x_input)
        # x = x_input

        ### Gating

        x = self.output_proj(x)
        gate = torch.sigmoid(x[:, 0])
        x = x[:, 0] * gate + (1 - gate) * x[:, -1]
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class MyModel_without_VQ_Gate_Mix_Large(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator3 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator4 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # img_y = torch.nn.functional.softmax(img, dim=-1)

        img_y = img

        
        # img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        # print(img_y[0,0], img_y[200,0])
        # print("*" * 20)
        img_z = img_y.argmax(dim=-1)
        return img_z

        # img_embed = self.img_embed(img_z)
        # x = self.output_proj(img_embed.squeeze(1))
        # return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)

        ### Softmax added
        # img = F.softmax(img, dim=-1)

        img_embed = torch.matmul(img, self.img_embed.weight)
        
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)
        x_4_agg = self.aggregator(x_3_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        # x_2 = F.batch_norm(x_2, running_mean=None, running_var=None, training=self.training)
        x_2 = F.relu(x_2)
        x_3 = self.aggregator2(x_2, edge_index)
        # x_3 = F.batch_norm(x_3, running_mean=None, running_var=None, training=self.training)
        x_3 = F.relu(x_3)
        x_4 = self.aggregator3(x_3, edge_index)
        # x_4 = F.batch_norm(x_4, running_mean=None, running_var=None, training=self.training)
        # x_4 = F.relu(x_4)
        # x_5 = self.aggregator4(x_4, edge_index)
        
        
        # img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x_4, x_2_agg, x_3_agg, x_4_agg), dim=1) # token_representation, 
        
        
        # x = self.transformer(x_input)
        # return x[:, 0]



        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer(x_input)

        ### just hyperparam. 

        # x = self.output_proj(x)
        # # gate = torch.sigmoid(x[:, 0])
        # # x_temp = x[:, 0] * gate + (1 - gate) * x[:, 3]
        # x = x[:, 0] * 0.99 + 0.01 * x[:, -1]
        # return x

        ### Gating

        x = self.output_proj(x)
        gate = torch.sigmoid(x[:, 0])
        x = x[:, 0] * gate + (1 - gate) * x[:, -1]
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class MyModel_with_VQ_Hetero_squirrel(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.tanh(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z

        img_embed = self.img_embed(img_z)
        x = self.output_proj(img_embed.squeeze(1))
        return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        # img = F.dropout(img, p=0.2, training=self.training) ## added
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)

        
        # x_2_agg += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        # x_3_agg += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        


        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # x_2 = torch.nn.functional.normalize(x_2, p=2, dim=1)
        # x_3 = torch.nn.functional.normalize(x_3, p=2, dim=1)
        
        # x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=1)

        # x += self.hop_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.hop_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x.device)+1)
        # x_3 += self.hop_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x.device)+2)
        # img_embed[:, 0] += self.hop_emb(torch.zeros(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 1] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 2] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device)+1)

        # prefix = self.prefix_token(torch.arange(10, device=x.device))
        # prefix = prefix.unsqueeze(0).expand(x.size(0), -1, -1)

        x_input = torch.stack((x_3, x_2_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x, x_3_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = x_3.unsqueeze(1) # token_representation, 
    

        # x_input = x_2.unsqueeze(1)
        # x_input = torch.cat((x_input, img_embed), dim=1)
        x_input = torch.cat((img_embed, x_input), dim=1)
        # x_input = torch.cat((x_input, img_embed, prefix), dim=1)
        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)
        x = self.output_proj(x)


        # x = self.output_proj(img_embed.squeeze(1))
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class MyModel_with_VQ_Hetero_squirrel_struct(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.tanh(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z

        img_embed = self.img_embed(img_z)
        x = self.output_proj(img_embed.squeeze(1))
        return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        # img = F.dropout(img, p=0.2, training=self.training) ## added
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)

        
        # x_2_agg += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        # x_3_agg += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        


        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # x_2 = torch.nn.functional.normalize(x_2, p=2, dim=1)
        # x_3 = torch.nn.functional.normalize(x_3, p=2, dim=1)
        
        # x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=1)

        # x += self.hop_emb(torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # x_2 += self.hop_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x.device)+1)
        # x_3 += self.hop_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x.device)+2)
        # img_embed[:, 0] += self.hop_emb(torch.zeros(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 1] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))
        # img_embed[:, 2] += self.hop_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device)+1)

        # prefix = self.prefix_token(torch.arange(10, device=x.device))
        # prefix = prefix.unsqueeze(0).expand(x.size(0), -1, -1)

        x_input = torch.stack((x_3, x_2_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x, x_3_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = x_3.unsqueeze(1) # token_representation, 
    

        # x_input = x_2.unsqueeze(1)
        # x_input = torch.cat((x_input, img_embed), dim=1)
        # x_input = torch.cat((img_embed, x_input), dim=1)
        # x_input = torch.cat((x_input, img_embed, prefix), dim=1)
        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)
        x = self.output_proj(x)


        # x = self.output_proj(img_embed.squeeze(1))
        return x[:,0]

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x

class PaCEr(nn.Module):
    def __init__(self, input_dim, class_dim, node_num,  mode='homo'):
        super().__init__()
        self.left_emb = nn.Embedding(node_num, input_dim)
        self.right_emb = nn.Embedding(node_num, input_dim)
        self.weighted_agg = WeightedAggregator()
        self.node_num = node_num
        self.linear = nn.Linear(input_dim, class_dim)
        if mode =='homo':
            self.gnn = torch_geometric.nn.GCNConv(input_dim, input_dim)
        else:
            self.gnn = torch_geometric.nn.GCNConv(input_dim + node_num, input_dim)
            
        self.mode = mode

    def forward(self, x, edge_index):
        left = self.left_emb(torch.arange(self.node_num, device=x.device))
        right = self.right_emb(torch.arange(self.node_num, device=x.device))
        out = torch.matmul(left, right.t())

        adj = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float32, device=x.device)
        row, col = edge_index
        adj[row, col] = 1.0
        # out = torch.where(adj == 0, torch.tensor(float('-inf'), device=adj.device), out)
        out_softmax = out * adj
        out_softmax = torch.where(out==0, torch.tensor(float('-inf'), device=adj.device), out_softmax)

        out = F.softmax(out_softmax, dim=-1)
        if self.mode == 'homo':
            edge_weight = out[row, col]
            node_emb = self.gnn(x, edge_index, edge_weight=edge_weight) 
            # node_emb = self.weighted_agg(x, edge_index, edge_weight)

            node_emb = F.relu(node_emb)
            node_emb = self.linear(node_emb)
            return node_emb, out
        else:
            x = torch.cat((x, out), dim=-1)
            node_emb = self.gnn(x,edge_index)
            node_emb = F.relu(node_emb)
            node_emb = self.linear(node_emb)
            return node_emb, out

class WeightedAggregator(MessagePassing):
    def __init__(self):
        # We do a sum aggregation
        super(WeightedAggregator, self).__init__(aggr='add')
    
    def forward(self, x, edge_index, weight_matrix):
        """
        x: Node features, shape [N, F]
        edge_index: Graph edges (2, E) in PyG format
        weight_matrix: Tensor [N, N], weight_matrix[i, j] is weight of edge (i,j)
        """
        # Extract edge weights from weight_matrix based on edge_index
        # row, col = edge_index
        # edge_weight = weight_matrix[row, col]

        # Start message passing
        return self.propagate(edge_index, x=x, edge_weight=weight_matrix)
    
    def message(self, x_j, edge_weight):
        # Message from neighbor j → i is scaled by edge weight
        return edge_weight.unsqueeze(-1) * x_j




### For the large dataset (KDD)

class MyModel_large_KDD(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator3 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

        # self.check_gcn1 = torch_geometric.nn.GCNConv(input_dim, embed_dim)
        # self.check_gcn2 = torch_geometric.nn.GCNConv(embed_dim, class_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # img_y = torch.nn.functional.softmax(img, dim=-1)

        img_y = img

        
        # img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        # print(img_y[0,0], img_y[200,0])
        # print("*" * 20)
        img_z = img_y.argmax(dim=-1)
        return img_z

        # img_embed = self.img_embed(img_z)
        # x = self.output_proj(img_embed.squeeze(1))
        # return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        # x = self.check_gcn1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.check_gcn2(x, edge_index)
        # return x



        x = self.input_proj(x)

        ## making image embedding with dataloaders


        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)

        ### Softmax added
        # img = F.softmax(img, dim=-1)

        img_embed = torch.matmul(img, self.img_embed.weight)
        
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)
        x_4_agg = self.aggregator(x_3_agg, edge_index)
        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)
        x_4 = self.aggregator3(x_3, edge_index)
        
        
        # img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x_4, x_2_agg, x_3_agg, x_4_agg), dim=1) # token_representation, 
        x_input = torch.stack((x_4, x_2, x_3, x), dim=1) # token_representation, 
        
        # return x_3
        # x = self.transformer(x_input)
        # return x[:, 0]


        x_input = torch.cat((x_input, img_embed), dim=1)
        # x_input = img_embed
        x = self.transformer(x_input)

        ### Gating

        x = self.output_proj(x)
        gate = torch.sigmoid(x[:, 0])
        x = x[:, 0] * gate + (1 - gate) * x[:, -1]
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x


class MyModel_large_KDD_diff(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, model_name, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()

        self.model_name = model_name
        if model_name == 'gcn':
            self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
            self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
            self.aggregator3 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        elif model_name == 'sage':
            self.aggregator1 = torch_geometric.nn.SAGEConv(embed_dim, embed_dim)
            self.aggregator2 = torch_geometric.nn.SAGEConv(embed_dim, embed_dim)
            self.aggregator3 = torch_geometric.nn.SAGEConv(embed_dim, embed_dim)
        elif model_name == 'gat':
            self.aggregator1 = torch_geometric.nn.GATConv(embed_dim, embed_dim//4, heads=4)
            self.aggregator2 = torch_geometric.nn.GATConv(embed_dim, embed_dim//4, heads=4)
            self.aggregator3 = torch_geometric.nn.GATConv(embed_dim, embed_dim//4, heads=4)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

        # self.check_gcn1 = torch_geometric.nn.GCNConv(input_dim, embed_dim)
        # self.check_gcn2 = torch_geometric.nn.GCNConv(embed_dim, class_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # img_y = torch.nn.functional.softmax(img, dim=-1)

        img_y = img

        
        # img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        # print(img_y[0,0], img_y[200,0])
        # print("*" * 20)
        img_z = img_y.argmax(dim=-1)
        return img_z

        # img_embed = self.img_embed(img_z)
        # x = self.output_proj(img_embed.squeeze(1))
        # return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img, mode='homo'):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        # x = self.check_gcn1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.check_gcn2(x, edge_index)
        # return x



        x = self.input_proj(x)

        ## making image embedding with dataloaders

        
        if mode == 'homo':
            img = self.input_proj_img1(img)
            img = F.relu(img)
            img = self.input_proj_img2(img)

            ### Softmax added
            # img = F.softmax(img, dim=-1)

            img_embed = torch.matmul(img, self.img_embed.weight)

        else:
            img = self.input_proj_img1(img)
            img = F.relu(img)
            # img = F.dropout(img, p=0.2, training=self.training) ## added
            img = self.input_proj_img2(img)
            img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
            img_z = img_y.argmax(dim=-1)
            img_embed = self.img_embed(img_z)
            # img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        
        
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)
        # x_4_agg = self.aggregator(x_3_agg, edge_index)
        

        x_2 = self.aggregator1(x, edge_index)
        x_2 = F.relu(x_2)
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_3 = self.aggregator2(x_2, edge_index)
        # x_4 = self.aggregator3(x_3, edge_index)
        
        
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x_4, x_2_agg, x_3_agg, x_4_agg), dim=1) # token_representation, 
        if mode == 'homo':
            x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        else:
            x_input = torch.stack((x_3, x_2_agg), dim=1) # token_representation, 
        
        # return x_3
        # x = self.transformer(x_input)
        # return x[:, -1]

        # Normalize x_input and img_embed to have the same norm
        
        if self.model_name == 'sage':
            
            x_input = torch.nn.functional.normalize(x_input, p=2, dim=-1)
            img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=-1)
        # print(f"x_input norm: {torch.sum(torch.norm(x_3, dim=-1))}, img_embed norm: {torch.sum(torch.norm(img_embed[:,2], dim=-1))}")
        x_input = torch.cat((x_input, img_embed), dim=1)
        # x_input = img_embed
        x = self.transformer(x_input)
        # return x[:, -1]
        ### Gating

        x = self.output_proj(x)
        # return x[:, -1]
        gate = torch.sigmoid(x[:, 0])
        x = x[:, 0] * gate + (1 - gate) * x[:, -1]
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x


class MyModel_large_KDD_squirrel(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.tanh(img)
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        return img_z

        img_embed = self.img_embed(img_z)
        x = self.output_proj(img_embed.squeeze(1))
        return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        x = self.input_proj(x)
        img = self.input_proj_img1(img)
        img = F.relu(img)
        # img = F.dropout(img, p=0.2, training=self.training) ## added
        img = self.input_proj_img2(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)

        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)

        
        img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))


        x_input = torch.stack((x_3, x_2_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x, x_3_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = x_3.unsqueeze(1) # token_representation, 
    

        # x_input = x_2.unsqueeze(1)
        # x_input = torch.cat((x_input, img_embed), dim=1)
        x_input = torch.cat((img_embed, x_input), dim=1)
        # x_input = torch.cat((x_input, img_embed, prefix), dim=1)
        # print(img_z.shape, x_input.shape, img_embed.shape)
        # exit(0)
        x = self.transformer(x_input)
        x = self.output_proj(x)


        # x = self.output_proj(img_embed.squeeze(1))
        # return x[:, 2]

        gate = torch.sigmoid(x[:, 0])
        x = x[:, 0] * gate + (1 - gate) * x[:, -1]
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)

        # return x[:, 2]

        gate = torch.sigmoid(x[:, 2])
        x = x[:, 2] * gate + (1 - gate) * x[:, -1]
        return x






import math
import numpy as np

import torch_geometric
import torch_geometric.utils as utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SAGE_LLPE(nn.Module):
    '''
    SAGE with Learnable Laplacian Position Encodings
    '''
    def __init__(self, in_dim, pos_dim=8, h_dim=32, out_dim=1, K=4, num_layers=2, dropout=0):
        super(SAGE_LLPE, self).__init__()
        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feat_enc = nn.Linear(in_dim, h_dim)

        # Chebyshev polynomial parameters
        self.K = K # order of Chebyshev polynomial
        self.alpha = torch.nn.parameter.Parameter(torch.empty((self.K, pos_dim), requires_grad=True), requires_grad=True) # coefficients of Chebyshev Polynomials
        nn.init.uniform_(self.alpha, -1/math.sqrt(self.K), 1/math.sqrt(self.K))

        self.hidden_encs = nn.ModuleList()
        for l in range(num_layers):
            self.hidden_encs.append(SAGEConv(pos_dim+h_dim, pos_dim+h_dim))
            self.hidden_encs.append(nn.ReLU())
            self.hidden_encs.append(nn.Dropout(p=dropout))
            
        self.readout_layer = nn.Linear(pos_dim+h_dim, out_dim)
        # self.readout_layer = nn.Linear(h_dim, out_dim)

    def forward(self, data, device):
        x = data.x
        
        # process eigenvectors: min-max normalization in range [-1, +1]
        eigenvectors = data.eigenvectors
        eig_min = torch.min(eigenvectors, dim=0, keepdim=True)[0]
        eig_max = torch.max(eigenvectors, dim=0, keepdim=True)[0]
        eigenvectors = (eigenvectors - eig_min)/(eig_max - eig_min + 1e-10) * (1 + 1) - 1 

        # process eigenvalues: scale eigenvalues of Laplacian from [0, 2] --> [-1, +1]
        eigenvalues = data.eigenvalues
        eig_max = torch.max(eigenvalues)
        eig_min = torch.min(eigenvalues)
        eigenvalues = (eigenvalues - eig_min)/(eig_max - eig_min + 1e-10) * (1 + 1) - 1

        # vectorized form of Chebyshev polynomial
        eigenvalues = torch.arccos(torch.broadcast_to(eigenvalues[:, None], (eigenvalues.shape[0], self.K)))
        eigenvalues = torch.mul(eigenvalues, torch.arange(self.K).to(device))
        eigenvalues = torch.cos(eigenvalues)
        eigenvalues = torch.matmul(eigenvalues, self.alpha)
        
        # pos encoding is the eigenvectors multiplied by chebyshev filters
        pos = torch.matmul(eigenvectors, eigenvalues)
        x = torch.cat([self.feat_enc(x), pos], dim=-1)
        # x = torch.cat([x, pos], dim=-1)

        edge_index = data.edge_index
        for l, layer in enumerate(self.hidden_encs[:-1]): 
            if np.mod(l, 3) == 0: 
                x = layer(x, edge_index) # GCN layer
            else: 
                x = layer(x) # dropout and relu layers

        x = self.readout_layer(x)
        # x = F.log_softmax(x, dim=-1)
        
        return x, eigenvalues

class GCN_LLPE(nn.Module):
    '''
    GCN with Learnable Laplacian Position Encodings
    '''
    def __init__(self, in_dim, pos_dim=8, h_dim=32, out_dim=1, K=64, num_layers=2, dropout=0):
        super(GCN_LLPE, self).__init__()
        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feat_enc = nn.Linear(in_dim, h_dim)

        # Chebyshev polynomial parameters
        self.K = K # order of Chebyshev polynomial
        self.alpha = torch.nn.parameter.Parameter(torch.empty((self.K, pos_dim), requires_grad=True), requires_grad=True) # coefficients of Chebyshev Polynomials
        nn.init.uniform_(self.alpha, -1/math.sqrt(self.K), 1/math.sqrt(self.K))

        self.hidden_encs = nn.ModuleList()
        for l in range(num_layers):
            self.hidden_encs.append(GCNConv(pos_dim+h_dim, pos_dim+h_dim))
            self.hidden_encs.append(nn.ReLU())
            self.hidden_encs.append(nn.Dropout(p=dropout))
            
        self.readout_layer = nn.Linear(pos_dim+h_dim, out_dim)
        # self.readout_layer = nn.Linear(h_dim, out_dim)

    def forward(self, data, device):
        x = data.x
        
        # process eigenvectors: min-max normalization in range [-1, +1]
        eigenvectors = data.eigenvectors
        eig_min = torch.min(eigenvectors, dim=0, keepdim=True)[0]
        eig_max = torch.max(eigenvectors, dim=0, keepdim=True)[0]
        eigenvectors = (eigenvectors - eig_min)/(eig_max - eig_min + 1e-10) * (1 + 1) - 1 

        # process eigenvalues: scale eigenvalues of Laplacian from [0, 2] --> [-1, +1]
        eigenvalues = data.eigenvalues
        eig_max = torch.max(eigenvalues)
        eig_min = torch.min(eigenvalues)
        eigenvalues = (eigenvalues - eig_min)/(eig_max - eig_min + 1e-10) * (1 + 1) - 1

        # vectorized form of Chebyshev polynomial
        eigenvalues = torch.arccos(torch.broadcast_to(eigenvalues[:, None], (eigenvalues.shape[0], self.K)))
        eigenvalues = torch.mul(eigenvalues, torch.arange(self.K).to(device))
        eigenvalues = torch.cos(eigenvalues)
        eigenvalues = torch.matmul(eigenvalues, self.alpha)
        
        # pos encoding is the eigenvectors multiplied by chebyshev filters
        pos = torch.matmul(eigenvectors, eigenvalues)
        x = torch.cat([self.feat_enc(x), pos], dim=-1)
        # x = torch.cat([x, pos], dim=-1)

        edge_index = data.edge_index
        for l, layer in enumerate(self.hidden_encs[:-1]): 
            if np.mod(l, 3) == 0: 
                x = layer(x, edge_index) # GCN layer
            else: 
                x = layer(x) # dropout and relu layers

        x = self.readout_layer(x)
        x = F.log_softmax(x, dim=-1)
        
        return x, eigenvalues


class GAT_LLPE(nn.Module):
    '''
    SAGE with Learnable Laplacian Position Encodings
    '''
    def __init__(self, in_dim, pos_dim=8, h_dim=32, out_dim=1, K=64, num_layers=2, dropout=0):
        super(GAT_LLPE, self).__init__()
        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feat_enc = nn.Linear(in_dim, h_dim)

        # Chebyshev polynomial parameters
        self.K = K # order of Chebyshev polynomial
        self.alpha = torch.nn.parameter.Parameter(torch.empty((self.K, pos_dim), requires_grad=True), requires_grad=True) # coefficients of Chebyshev Polynomials
        nn.init.uniform_(self.alpha, -1/math.sqrt(self.K), 1/math.sqrt(self.K))

        self.hidden_encs = nn.ModuleList()
        for l in range(num_layers):
            self.hidden_encs.append(torch_geometric.nn.GATConv(pos_dim+h_dim, pos_dim+h_dim, heads=1, dropout=dropout))
            self.hidden_encs.append(nn.ReLU())
            self.hidden_encs.append(nn.Dropout(p=dropout))
            
        self.readout_layer = nn.Linear(pos_dim+h_dim, out_dim)
        # self.readout_layer = nn.Linear(h_dim, out_dim)

    def forward(self, data, device):
        x = data.x
        
        # process eigenvectors: min-max normalization in range [-1, +1]
        eigenvectors = data.eigenvectors
        eig_min = torch.min(eigenvectors, dim=0, keepdim=True)[0]
        eig_max = torch.max(eigenvectors, dim=0, keepdim=True)[0]
        
        eigenvectors = (eigenvectors - eig_min)/(eig_max - eig_min + 1e-10) * (1 + 1) - 1 

        # process eigenvalues: scale eigenvalues of Laplacian from [0, 2] --> [-1, +1]
        eigenvalues = data.eigenvalues
        eig_max = torch.max(eigenvalues)
        eig_min = torch.min(eigenvalues)
        eigenvalues = (eigenvalues - eig_min)/(eig_max - eig_min + 1e-10) * (1 + 1) - 1

        # vectorized form of Chebyshev polynomial
        eigenvalues = torch.arccos(torch.broadcast_to(eigenvalues[:, None], (eigenvalues.shape[0], self.K)))
        eigenvalues = torch.mul(eigenvalues, torch.arange(self.K).to(device))
        eigenvalues = torch.cos(eigenvalues)
        eigenvalues = torch.matmul(eigenvalues, self.alpha)
        
        # pos encoding is the eigenvectors multiplied by chebyshev filters
        pos = torch.matmul(eigenvectors, eigenvalues)
        x = torch.cat([self.feat_enc(x), pos], dim=-1)
        # x = torch.cat([x, pos], dim=-1)

        edge_index = data.edge_index
        for l, layer in enumerate(self.hidden_encs[:-1]): 
                # x = layer(x, edge_index) # GCN layer
            if np.mod(l, 3) == 0: 
                x = layer(x, edge_index) # GCN layer
            else: 
                x = layer(x) # dropout and relu layers

        x = self.readout_layer(x)
        # x = F.log_softmax(x, dim=-1)
        
        return x, eigenvalues


class MyModel_large_KDD_struct(nn.Module):
    def __init__(self, input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, image_dim=2048, img_embed_num=256):
        super().__init__()
        self.aggregator = SimpleAggregator()
        self.aggregator1 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator2 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.aggregator3 = torch_geometric.nn.GCNConv(embed_dim, embed_dim)
        self.transformer = Transformer(embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_proj_img1 = nn.Linear(image_dim, embed_dim)
        self.input_proj_img2 = nn.Linear(embed_dim, img_embed_num)
        self.img_embed = nn.Embedding(img_embed_num, embed_dim)
        self.output_proj = nn.Linear(embed_dim, class_dim)
        self.dropout = dropout
        self.token_representation = nn.Parameter(torch.randn(1, embed_dim))
        self.type_emb = nn.Embedding(2, embed_dim)
        self.hop_emb = nn.Embedding(3, embed_dim)
        # self.prefix_token = nn.Embedding(10, embed_dim)

        # self.check_gcn1 = torch_geometric.nn.GCNConv(input_dim, embed_dim)
        # self.check_gcn2 = torch_geometric.nn.GCNConv(embed_dim, class_dim)

    def get_img_label(self, img):
        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)
        # img_y = torch.nn.functional.softmax(img, dim=-1)

        img_y = img

        
        # img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        # print(img_y[0,0], img_y[200,0])
        # print("*" * 20)
        img_z = img_y.argmax(dim=-1)
        return img_z

        # img_embed = self.img_embed(img_z)
        # x = self.output_proj(img_embed.squeeze(1))
        # return x.argmax(dim=-1)
    
    def forward(self, x, edge_index, img):
        # token_representation = self.token_representation.expand(x.size(0), -1)
        # x = self.check_gcn1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.check_gcn2(x, edge_index)
        # return x



        x = self.input_proj(x)

        ## making image embedding with dataloaders


        img = self.input_proj_img1(img)
        img = F.relu(img)
        img = self.input_proj_img2(img)

        ### Softmax added
        # img = F.softmax(img, dim=-1)

        img_embed = torch.matmul(img, self.img_embed.weight)
        
        x_2_agg = self.aggregator(x, edge_index)
        x_3_agg = self.aggregator(x_2_agg, edge_index)
        x_4_agg = self.aggregator(x_3_agg, edge_index)
        

        x_2 = self.aggregator1(x, edge_index)
        x_3 = self.aggregator2(x_2, edge_index)
        x_4 = self.aggregator3(x_3, edge_index)
        
        
        # img_embed += self.type_emb(torch.ones((img_embed.size(0), img_embed.size(1)), dtype=torch.long, device=img_embed.device))

        # x_input = torch.stack((x_3, x_2_agg, x_3_agg), dim=1) # token_representation, 
        # x_input = torch.stack((x_4, x_2_agg, x_3_agg, x_4_agg), dim=1) # token_representation, 
        x_input = torch.stack((x_3, x_2, x), dim=1) # token_representation, 
        
        # return x_3
        # x = self.transformer(x_input)
        # return x[:, 0]


        x_input = torch.cat((x_input, img_embed), dim=1)
        # x_input = img_embed
        x = self.transformer(x_input)
        # return x[:, -1]
        ### Gating

        x = self.output_proj(x)
        return x[:, -1]
        gate = torch.sigmoid(x[:, 0])
        x = x[:, 0] * gate + (1 - gate) * x[:, -1]
        return x

    def median_act(self, x, edge_index, img, target):
        x = self.input_proj(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        img = self.input_proj_img(img)
        img_y = torch.nn.functional.gumbel_softmax(img, tau=1, hard=True)
        img_z = img_y.argmax(dim=-1)
        img_embed = self.img_embed(img_z)

        x_2 = self.aggregator(x, edge_index)
        x_3 = self.aggregator(x_2, edge_index)
        x_4 = self.aggregator(x_3, edge_index)
        # x_5 = self.aggregator(x_4, edge_index)
        # x_input = torch.stack((x, x_2), dim=1)
        # x_input = torch.stack((x, x_2, x_3), dim=1)

        x_2 += self.type_emb(torch.zeros(x_2.size(0), dtype=torch.long, device=x_2.device))
        x_3 += self.type_emb(torch.zeros(x_3.size(0), dtype=torch.long, device=x_3.device))
        x_4 += self.type_emb(torch.zeros(x_4.size(0), dtype=torch.long, device=x_4.device))
        img_embed += self.type_emb(torch.ones(img_embed.size(0), dtype=torch.long, device=img_embed.device))

        x_input = torch.stack((x, x_2, x_3, x_4), dim=1)
        x_input = torch.cat((x_input, img_embed), dim=1)
        x = self.transformer.median_act(x_input, target)
        x = self.output_proj(x)
        return x
