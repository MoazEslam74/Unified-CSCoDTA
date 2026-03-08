import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, smi_dim, seq_dim, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(smi_dim, hid_dim)
        self.w_k = nn.Linear(seq_dim, hid_dim)
        self.w_v = nn.Linear(seq_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        #[batch size, sent len, hid dim]
        x = x.permute(0, 2, 1) #[batch size, hid dim, sent len]
        x = self.do(F.relu(self.fc_1(x)))# [batch size, pf dim, sent len]
        x = self.fc_2(x) # [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1) #[batch size, sent len, hid dim]
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, hid_dim, smi_dim, seq_dim, n_heads, dropout=0.2, device="cuda"):
        super().__init__()
        hid_dim = smi_dim
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = CrossAttentionLayer(smi_dim, smi_dim, hid_dim, n_heads, dropout, device)
        self.ea = CrossAttentionLayer(smi_dim, seq_dim, hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, seq_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, smi, seq, smi_mask=None, seq_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        smi = self.ln(smi + self.do(self.sa(smi, smi, smi, smi_mask)))

        smi = self.ln(smi + self.do(self.ea(smi, seq, seq, seq_mask)))

        smi = self.ln(smi + self.do(self.pf(smi)))

        return smi

# ---------- New Classes for Unified Framework ---------- #

class SymmetricFusion(nn.Module):
    """
    Computes symmetric features from two drug embeddings.
    Combines Hadamard product (element-wise multiplication) 
    and absolute difference to maintain symmetry.
    """
    def __init__(self):
        super().__init__()

    def forward(self, drug1_emb, drug2_emb):
        # drug1_emb, drug2_emb shape: [batch_size, features] or [batch_size, seq_len, features]
        
        # Hadamard product
        hadamard = drug1_emb * drug2_emb
        
        # Absolute difference
        abs_diff = torch.abs(drug1_emb - drug2_emb)
        
        # Concatenate along the last dimension (features)
        fusion_feat = torch.cat((hadamard, abs_diff), dim=-1)
        
        return fusion_feat


class DDI_Decoder(nn.Module):
    """
    Decoder specifically for Drug-Drug Interaction (DDI) prediction.
    """
    def __init__(self, args):
        super().__init__()
        self.smi_hidden_size = args["smi_hidden_size"]
        
        # The input dimension is doubled because SymmetricFusion concatenates two tensors
        fusion_dim = self.smi_hidden_size * 2 
        
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # Output a single logit for binary classification
        self.relu = nn.ReLU()
        # Note: We omit Sigmoid here as it's better to use BCEWithLogitsLoss during training

    def forward(self, fusion_feat):
        # fusion_feat shape: [batch_size, fusion_dim]
        x = self.relu(self.fc1(fusion_feat))
        x = self.relu(self.fc2(x))
        logit = self.fc3(x)
        return logit

# ------------------------------------------------------- #

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args["device"]
        self.att_dim = args["att_dim"]
        self.n_layers = args["n_layers"]
        self.n_heads = args["n_heads"]
        self.smi_hidden_size = args["smi_hidden_size"]
        self.seq_hidden_size = args["seq_hidden_size"]
        self.cross_attention = nn.ModuleList(
            [
                CrossAttention(self.att_dim, self.smi_hidden_size, self.seq_hidden_size, self.n_heads)
                for _ in range(self.n_layers)
            ]
        )
        self.fc1 = nn.Linear(self.smi_hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.predict_layer = nn.PReLU()
    
    def forward(self, smi, seq):
        for layer in self.cross_attention:
            smi = layer(smi, seq)
        x = torch.norm(smi, dim=2)
        x = F.softmax(x, dim=1)
        fusion = torch.zeros((smi.shape[0], self.smi_hidden_size)).to(self.device)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                value = smi[i, j]
                value *= x[i, j]
                fusion[i] += value
        
        label = self.fc1(fusion)

        # label = F.relu(self.fc1(fusion))
        # label = self.fc2(label)
        # label = self.predict_layer(label)
        return label