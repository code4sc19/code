import torch, math, config
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
conf = config.config()
#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.d_k = emb_dim / heads
        self.heads = heads
        self.q_linear = nn.Linear(emb_dim, emb_dim).to(device)
        self.v_linear = nn.Linear(emb_dim, emb_dim).to(device)
        self.k_linear = nn.Linear(emb_dim, emb_dim).to(device)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, emb_dim).to(device)
        #alpha is the scalable parameter
        self.alpha = Variable(torch.rand(1, heads, 1, 1)).to(device)
        self.params_init(self.q_linear.named_parameters())
        self.params_init(self.v_linear.named_parameters())
        self.params_init(self.k_linear.named_parameters())
        self.params_init(self.out.named_parameters())
                
    def params_init(self, params):
        #kaiming initialization is used
        for name, param in params:
            if len(param.data.shape)==2:
                nn.init.kaiming_normal_(param, a=1, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal_(param)

    def attention(self, q, k, v, d_k, mask, dep, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        bz = scores.size()[0]
        seq_l = scores.size()[-2]
        mask = LongTensor(mask)
        if mask is not None:
            mask = mask.repeat(1, self.heads, 1).view(bz, self.heads, seq_l, seq_l)
            scores = scores.masked_fill(mask==0, -1e9)
            dep=np.expand_dims(dep,1)
            dep=np.repeat(dep, self.heads, axis=1)
            dep=FloatTensor(dep)
            assert dep.shape==scores.shape
            #scores = F.softmax(torch.add(scores, dep*self.alpha), dim=-1)
            scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores, v)
        return output, scores
    
    def forward(self, q, k, v, mask, dep):
        """
        Args:
        	q: Queries tensor: [bz, seq_len, emb_dim]
        	k: Keys tensor:    [bz, seq_len, emb_dim]
        	v: Values tensor:  [bz, seq_len, emb_dim], in general, k serves as v
        	attn_mask: Masking tensor with shape [bz, seq_len, seq_len]

        Returns:
        	masked z
        """
        att_mask=np.expand_dims(mask, axis=-1)
        att_mask=att_mask*att_mask.transpose(0, 2, 1)
        bz = q.size(0)
     
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bz, -1, self.heads, self.d_k)
        q = self.q_linear(q).view(bz, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(bz, -1, self.heads, self.d_k)
        
        # transpose to get dimensions bz * heads * seq_len * emb_dim_per_head
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we define
        z, att_scores = self.attention(q, k, v, self.d_k, att_mask, dep, self.dropout)
        # concatenate heads and put through final linear layer
        shape = z.shape
        z = z.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.heads)
        z = self.out(z)
        
        return z*FloatTensor(mask).unsqueeze(-1), att_scores

class FeedForward(nn.Module):
    def __init__(self, emb_dim, d_ff=2048, dropout = 0.1):
        super(FeedForward, self).__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(emb_dim, d_ff).to(device)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, emb_dim).to(device)
        self.params_init(self.linear_1.named_parameters())
        self.params_init(self.linear_2.named_parameters())

    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                nn.init.kaiming_normal_(param, a=1, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal_(param)

    def forward(self, x, mask):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x*FloatTensor(mask).unsqueeze(-1)

#%%
class PositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_seq_len):
        super(PositionalEncoder, self).__init__()
        self.emb_dim = emb_dim
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, emb_dim)
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                pe[pos, i]     = math.sin(pos / (10000 ** ((2 * i)      /float(emb_dim))))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/float(emb_dim))))  
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.emb_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x_pe = Variable(self.pe[:,:seq_len], requires_grad=False).to(device)
        x = x + x_pe
        return x

class Norm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-18):
        super(Norm, self).__init__()
        self.size = emb_dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, heads, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        #self.norm_1 = Norm(emb_dim)
        #self.norm_2 = Norm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, heads)
        self.ff = FeedForward(emb_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
  	
    def forward(self, x, mask, dep):
        z, att_scores = self.attn(x,x,x,mask,dep)
        x = x + self.dropout_1(z)
        x = x + self.dropout_2(self.ff(x, mask))
        return x, att_scores   

import copy
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
 
class Encoder(nn.Module):
    def __init__(self, emb_dim, heads, num_layers, num_classes=2):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.pe = PositionalEncoder(emb_dim, max_seq_len=63)
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(emb_dim, heads)) for i in range(num_layers)]) 
        #self.norm = Norm(emb_dim)
        self.att = MultiHeadAttention(emb_dim, heads)
        self.fnn = FeedForward(emb_dim)
        self.linear_w = nn.Linear(emb_dim, num_classes).to(device)
        self.params_init(self.linear_w.named_parameters())
       
    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                nn.init.kaiming_normal_(param, a=1, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal_(param)

    def forward(self, x, mask, dep):
        att_list=[]
        x = self.pe(x)
        x = x*FloatTensor(mask).unsqueeze(-1)
        for i in range(self.num_layers):
            x, att_scores = self.layers[i](x, mask, dep)
            x = x*FloatTensor(mask).unsqueeze(-1)
	    att_list.append(att_scores)
        #x = self.norm(x)
        x = self.linear_w(x)
        return x, att_list
