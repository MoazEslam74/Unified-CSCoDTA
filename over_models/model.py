import torch
import torch.nn as nn
from torch.nn import functional as F

# قمنا بتحديث الاستيراد ليشمل الكلاسات الجديدة التي تمت إضافتها
from over_models.decoder import Decoder, SymmetricFusion, DDI_Decoder

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features  
        self.dropout = dropout 
        self.alpha = alpha  
        self.concat = concat  

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.matmul(inp, self.W)
        bs = h.size()[0] 
        N = h.size()[1]  
        adj = torch.eye(h.size()[1]).unsqueeze(0)
        adj = adj.repeat(bs, 1, 1).to(h.device)
        a_input = torch.cat([h.repeat(1, 1, N).view(bs, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(bs, N, -1, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -1e12 * torch.ones_like(e) 
        attention = torch.where(adj > 0, e, zero_vec)  

        attention = F.softmax(attention, dim=-1)  
        attention = F.dropout(attention, self.dropout, training=self.training) 
        h_prime = torch.matmul(attention, h) 

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.2, alpha=0.2, nheads=1):
        super(GAT, self).__init__()
        self.dropout = dropout
 
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
 
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x)  
        return x

class SMI_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.smi_hidden_size = args["smi_hidden_size"]
        self.smi_size = args["max_smi_len"]
        self.smi_embed = nn.Linear(self.smi_size, self.smi_hidden_size)
        self.gat = GAT(self.smi_hidden_size, self.smi_hidden_size)
    
    def forward(self, x, adj=None):
        x = self.smi_embed(x)
        x = self.gat(x, adj)
        return x

class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DilatedParllelResidualBlock(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  
        self.d2 = CDilated(n, n, 3, 1, 2)  
        self.d4 = CDilated(n, n, 3, 1, 4)  
        self.d8 = CDilated(n, n, 3, 1, 8)  
        self.d16 = CDilated(n, n, 3, 1, 16) 
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)
        
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class SEQ_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_hidden_size = args["seq_hidden_size"]
        self.seq_size = args["seq_size"]
        self.seq_embed = nn.Linear(self.seq_size, self.seq_hidden_size)
        
        convs = []
        convs_sizes = [64, 128, 256]
        temp_size = self.seq_hidden_size
        for conv_size in convs_sizes:
            convs.append(DilatedParllelResidualBlock(temp_size, conv_size))
            temp_size = conv_size
        self.dilated_conv = nn.Sequential(*convs)

    def forward(self, x):
        x = self.seq_embed(x)
        x = x.transpose(1, 2)
        x = self.dilated_conv(x)
        x = x.squeeze()
        return x 

# ========================================================================= #
# Unified Model (DTI & DDI)
# ========================================================================= #
class New(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 1. المشفرات (Encoders)
        self.smi_encoder = SMI_Encoder(self.args) # Shared Drug Encoder (للأوزان المشتركة)
        self.seq_encoder = SEQ_Encoder(self.args) # Target Encoder (للبروتين فقط)

        # 2. رؤوس المهام (Task-Specific Heads)
        # --- رأس DTI ---
        self.dti_decoder = Decoder(args) 
        
        # --- رأس DDI ---
        self.symmetric_fusion = SymmetricFusion()
        self.ddi_decoder = DDI_Decoder(args)

    def get_smi(self, smi):
        return smi
    
    def get_seq(self, seq):
        seq = self.seq_encoder(seq)
        seq = seq.transpose(0, 1)
        return seq

    def forward(self, task, input1, input2):
        """
        تقوم هذه الدالة بتوجيه البيانات حسب المهمة المطلوبة.
        - task: 'dti' أو 'ddi'
        - input1: الدواء الأول دائماً (Drug 1 Graph)
        - input2: البروتين (إذا كان task='dti') أو الدواء الثاني (إذا كان task='ddi')
        """
        if task == 'dti':
            # مسار تقارب الدواء والهدف (DTI/DTA)
            smi_emb = self.get_smi(input1)
            seq_emb = self.get_seq(input2)
            
            # الانتباه المتقاطع والتنبؤ
            score = self.dti_decoder(smi_emb, seq_emb)
            return score

        elif task == 'ddi':
            # مسار تفاعل دواء مع دواء (DDI)
            
            # تمرير الدواءين عبر نفس المشفر (Shared Weights)
            drug1_emb = self.get_smi(input1)
            drug2_emb = self.get_smi(input2)
            
            # الدمج المتماثل (Symmetric Fusion)
            fusion_feat = self.symmetric_fusion(drug1_emb, drug2_emb)
            
            # تصنيف التفاعل (Interaction Prediction)
            score = self.ddi_decoder(fusion_feat)
            return score
            
        else:
            raise ValueError("Task must be either 'dti' or 'ddi'.")