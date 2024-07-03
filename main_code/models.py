import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
    


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





class myTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(myTransformerEncoder, self).__init__()
        self.attention = SelfAttention(input_dim, hidden_dim)
        self.feedforward = ffn(hidden_dim, hidden_dim, output_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 添加残差连接和层归一化
        residual = x
        x = self.layer_norm1(x + self.dropout(self.attention(x)))
        x = self.layer_norm2(x + self.dropout(self.feedforward(x)))
        return x + residual  # 最后再添加一次残差连接

class featureEncoder(nn.Module):
    def __init__(self, dropout=0.5):
        super(featureEncoder, self).__init__()
        self.liner = nn.Linear(27, 64)
        self.transformer_1 = myTransformerEncoder(64, 64, 64, dropout)
        self.transformer_2 = myTransformerEncoder(64, 64, 64, dropout)
        self.feature_class = nn.Linear(64, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, feature_input):
        # Forward pass for text input
        feature_output = self.liner(feature_input)
        feature_output = self.transformer_1(feature_output)
        feature_output = self.transformer_2(feature_output)

        feature_feature = self.dropout(feature_output)

        feature_output = self.feature_class(feature_feature)

        return feature_output,feature_feature





class textEncoder(nn.Module):
    def __init__(self, path, dropout=0.5):
        super(textEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.text_class = nn.Linear(768, 5)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id, mask):
        #进行bert模型的正向传播计算
        _, text_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        text_feature = self.dropout(text_output)
        text_output = self.text_class(text_feature)
        return text_output,text_feature

class textEncoder(nn.Module):
    def __init__(self, path, dropout=0.5):
        super(textEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.text_class_1 = nn.Linear(768, 128)
        self.text_class_2 = nn.Linear(128, 5)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id, mask):
        #进行bert模型的正向传播计算
        _, text_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        text_feature = self.dropout(text_output)
        text_output = self.gelu(self.dropout(self.text_class_1(text_feature)))
        text_output = self.text_class_2(text_output)
        return text_output,text_feature
    
class textEncoder_2(nn.Module):
    def __init__(self, path, dropout=0.5):
        super(textEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.text_class_1 = nn.Linear(768, 128)
        self.text_class_2 = nn.Linear(128, 5)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id, mask):
        #进行bert模型的正向传播计算
        _, text_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        text_feature = self.dropout(text_output)
        text_feature_128 = self.gelu(self.dropout(self.text_class_1(text_feature)))
        text_output = self.text_class_2(text_feature_128)
        return text_output,(text_feature,text_feature_128)
    


# text_2
# from torch.nn import MultiheadAttention  
# class textEncoder(nn.Module):  
#     def __init__(self, path, dropout=0.5, num_heads=8, dim_feedforward=2048):  
#         super(textEncoder, self).__init__()  
#         self.bert = BertModel.from_pretrained(path)  
#         self.text_class_1 = nn.Linear(768, 128)  
#         self.text_class_2 = nn.Linear(128, 5) 
#         self.gelu = nn.GELU()  
#         self.dropout = nn.Dropout(dropout)  
  
#         # 添加多头自注意力层  
#         self.mha = MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout)  
#         self.linear_out = nn.Linear(768, 768)  
  
#     def forward(self, input_id, mask):  
#         # 进行BERT模型的正向传播计算  
#         text_output, poolling_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)  
  
#         # 添加多头自注意力层  
#         # 注意：这里为了简化，我们假设text_output的shape是[batch_size, sequence_length, embedding_dim]  
#         # 在实际应用中，可能需要调整query, key, value的shape  
#         query = key = value = text_output.permute(1, 0, 2)  
  
#         attention_output, attention_weights = self.mha(query, key, value)  
#         attention_output = attention_output.permute(1, 0, 2)  # [batch_size, sequence_length, embedding_dim]  
  
#         # 经过一个线性层和一个dropout  
#         attention_output = self.dropout(self.gelu(self.linear_out(attention_output))) 
  
#         # 对整个序列取平均（或其他聚合方法）以获得文本表示  
#         text_feature = attention_output.mean(dim=1)  
  
#         # 分类层  
#         text_output = self.dropout(self.gelu(self.text_class_1(text_feature))) 
#         text_output = self.text_class_2(text_output)  
  
#         return text_output, text_feature


    

    
class textEncoder_lstm(nn.Module):
    def __init__(self, path, dropout=0.5):
        super(textEncoder_lstm, self).__init__()
        # 加载 bert 模型
        self.bert = BertModel.from_pretrained(path)
        # 冻结 bert 参数
        for param in self.bert.parameters():
            param.requires_grad = False
        # 定义 dropout
        self.dropout = nn.Dropout(dropout)
        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size=768, hidden_size=384, num_layers=1, batch_first=True,bidirectional=True)
        self.linear = nn.Linear(384 * 2, 5)

    def forward(self, input_id, mask):
        # 进行 bert 模型的正向传播计算
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        # 进行 dropout 层的正向传播计算
        dropout_output = self.dropout(pooled_output)
        # LSTM层的输入应为三维张量，所以需要增加一个维度
        lstm_input = dropout_output.unsqueeze(1)
        # LSTM层的正向传播计算
        lstm_output, _ = self.lstm(lstm_input)
        # 取最后一个时间步的输出
        lstm_output = lstm_output[:, -1, :]
        # 进行线性层传播计算
        linear_output = self.linear(lstm_output)
        return linear_output,lstm_output