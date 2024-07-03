import torch
from torch import nn
import torch.nn.functional as F


class ffn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(ffn, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.feedforward(x)



class ffn_gelu(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(ffn_gelu, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.feedforward(x)








class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q,K,V):
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)
        attention_weights = torch.matmul(Q, K.transpose(0,1)) / torch.sqrt(torch.tensor(self.hidden_dim).float())
        attention_weights = self.softmax(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.attention = Attention(input_dim, hidden_dim)
        self.feedforward = ffn(hidden_dim, hidden_dim, output_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, K,Q,V):
        # 添加残差连接和层归一化
        x = self.layer_norm1(x + self.dropout(self.attention(K,Q,V)))
        x = self.layer_norm2(x + self.dropout(self.feedforward(x)))
        return x




class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = torch.matmul(Q, K.transpose(0,1)) / torch.sqrt(torch.tensor(self.hidden_dim).float())
        attention_weights = self.softmax(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output
    