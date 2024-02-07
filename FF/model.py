import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class GatedFusion(nn.Module):
    def __init__(self, D: int):
        super(GatedFusion, self).__init__()
        self._fully_connected_xs = nn.Linear(D, D, bias=False)
        self._fully_connected_xt = nn.Linear(D, D)
        self._fully_connected_h = nn.Linear(D, D)
        self.leaky_relu = nn.LeakyReLU()

    def forward(
            self, HS: torch.FloatTensor, HT: torch.FloatTensor
    ) -> torch.FloatTensor:
        XS = self._fully_connected_xs(HS)
        XT = self._fully_connected_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self._fully_connected_h(H)
        H = self.leaky_relu(H)
        del XS, XT, z
        return H


class SpatialEmbeddingAggregation(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, dropout=dropout)

    def forward(self, x):
        # x -> [300, 60, 64] -> [60, 300, 64]
        src = x.transpose(1, 0)
        src = self.pos_encoder(src)
        output, _ = self.mha(src, src, src)
        out = output[-1]
        out = torch.layer_norm(out, normalized_shape=[self.d_model])

        return out


class SSA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=True)

    def masked_softmax(self, x, mask, eps=-1e10):
        x = x.masked_fill(~mask.bool(), eps)
        x = torch.softmax(x, dim=-1).squeeze()
        return x

    def forward(self, x, edge=None):
        DEVICE = x.device
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (x.shape[-1] ** -0.5)

        value = torch.FloatTensor(np.ones(edge.shape[1])).to(DEVICE)

        graph = torch.sparse.FloatTensor(indices=edge, values=value, size=attn[0].shape).to_dense().to(DEVICE)

        mask_rh = torch.where(graph >= 1, 1, 0)

        attn = torch.mul(attn, mask_rh).to(DEVICE)
        attn = self.masked_softmax(attn, mask_rh).squeeze()

        v = torch.matmul(attn, v)

        return v, attn


class DMHSA(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=True)
        self.num_heads = num_heads
        self.seq_len = 60

    def masked_softmax(self, x, mask, eps=-1e10):
        x = x.masked_fill(~mask.bool(), eps)
        x = torch.softmax(x, dim=-1).squeeze()
        return x

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(2, 3)) * (x.shape[-1] ** -0.5)
        mask_attn = attn

        dynamic_mask = torch.where(mask_attn >= torch.mean(mask_attn, dim=-1, keepdim=True), 1, 0).reshape(B,
                                                                                                           self.num_heads,
                                                                                                           N, N)
        attn = torch.mul(attn, dynamic_mask)
        attn = self.masked_softmax(attn, dynamic_mask).squeeze()
        v = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)

        return v, attn


class SparseSDTransformerLayer(nn.Module):
    def __init__(self, d_model=20, n_heads=2, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.layer_norm_eps = 1e-5
        self.hidden_size = 64
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.mlp_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.dmhsa = DMHSA(dim=self.hidden_size, num_heads=n_heads)
        self.ssa = SSA(dim=self.hidden_size)

    def forward(self, x, edge):
        out = x
        dynamic_emb, mask = self.dmhsa(out)
        static_emb, _ = self.ssa(out, edge)
        value = dynamic_emb.squeeze() + static_emb.squeeze()
        residual = value + x
        out = self.layer_norm1(residual)
        out = self.mlp_out(out)
        out = self.layer_norm2(out + residual)
        return out, mask


class SparseSDTransformerEncoder(nn.Module):
    def __init__(self, d_feat, hidden_size, num_layers, batch_first=True, n_head=2, seq_len=60):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.n_head = n_head
        self.seq_len = seq_len
        self.sparse_trans1 = SparseSDTransformerLayer(d_model=d_feat, n_heads=n_head)
        self.relu = nn.LeakyReLU()

    def forward(self, x, edge_index):
        # [300,60,64]
        out = x
        out = out.transpose(1, 0)
        out, mask = self.sparse_trans1(out, edge_index)
        out = out.transpose(1, 0)
        out = self.relu(out)
        return out, mask


class TemporalEncoder(nn.Module):
    def __init__(self, d_feat=6, d_model=64, num_layers=2, dropout=0.5):
        super(TemporalEncoder, self).__init__()
        self.dropout = dropout
        self.rnn = nn.GRU(d_feat, d_model, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.relu = nn.LeakyReLU()

    def forward(self, src):
        out, _ = self.rnn(src)
        out = self.relu(out)
        return out.squeeze()


class Finformer(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, seq_len=60, temporal_dropout=0.0, snum_head=2, num_layers=2):
        super(Finformer, self).__init__()
        self.fc_in = nn.Linear(d_feat, hidden_size)
        self.ssdt = SparseSDTransformerEncoder(d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers,
                                               batch_first=True, n_head=snum_head, seq_len=seq_len)
        self.te = TemporalEncoder(d_feat=hidden_size, d_model=hidden_size, num_layers=num_layers,
                                  dropout=temporal_dropout)
        self.gate_fusion = GatedFusion(hidden_size)
        self.s_aggr = SpatialEmbeddingAggregation(d_model=hidden_size, dropout=temporal_dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index):
        x = self.fc_in(x)
        ts = self.te(x)
        res_ts = ts[:, -1, :].squeeze()

        # Static-Dynamic
        # [300,60,64]
        move, mask = self.ssdt(x, edge_index)
        out = self.s_aggr(move)
        out = self.gate_fusion(out, res_ts)
        out = self.linear(out)
        return out.squeeze(), mask.squeeze()
