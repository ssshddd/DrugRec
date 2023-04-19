import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    """Construct the embeddings from clinical token, position, segment embeddings.
    """
    def __init__(self, hidden_size, vocab_size, max_position_size, max_segment, word_emb_padding_idx, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=word_emb_padding_idx)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size, padding_idx=max_position_size-1)
        self.segment_embeddings = nn.Embedding(max_segment, hidden_size, padding_idx=max_segment-1)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_seq_len = max_position_size
        self.padding_idx_list = [word_emb_padding_idx, max_position_size-1, max_segment-1]

    def padding(self, input_ids, padding_idx):
        if input_ids.shape[1] < self.max_seq_len:
            input_ids = F.pad(input_ids[0], (0, self.max_seq_len - input_ids.shape[1]), 'constant', padding_idx).unsqueeze(0)
        padding_mask = (input_ids == padding_idx)
        return input_ids, padding_mask

    def forward(self, input_ids_list):
        input_ids_list = [idx[:,1:] if i>0 else idx for i, idx in enumerate(input_ids_list)]
        seq_length = [idx.shape[1] for idx in input_ids_list]
        position_ids = [torch.arange(seq_length[i], dtype=torch.long, device=input_ids_list[i].device) for i in range(len(seq_length))]
        position_ids = torch.cat([position_ids[i].unsqueeze(0) for i in range(len(position_ids))], dim=-1).expand_as(torch.cat(input_ids_list, dim=-1))
        segment_ids = [torch.zeros(seq_length[i], dtype=torch.long, device=input_ids_list[i].device)+k for k, i in enumerate(range(len(seq_length)))]
        segment_ids = torch.cat([segment_ids[i].unsqueeze(0) for i in range(len(segment_ids))], dim=-1).expand_as(torch.cat(input_ids_list, dim=-1))
        input_ids_list = torch.cat(input_ids_list, dim=-1)
        input_ids_list, padding_mask = self.padding(input_ids_list, self.padding_idx_list[0])
        position_ids,_ = self.padding(position_ids, self.padding_idx_list[1])
        segment_ids,_ = self.padding(segment_ids, self.padding_idx_list[2])

        words_embeddings = self.word_embeddings(input_ids_list)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, padding_mask


class Transformer_Encoder(nn.Module):
    def __init__(self, n_layer, hidden_size, num_attention_heads, vocab_size, max_position_size, max_segment, word_emb_padding_idx, dropout):
        super(Transformer_Encoder, self).__init__()
        self.emb = Embeddings(hidden_size, vocab_size, max_position_size, max_segment, word_emb_padding_idx, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=hidden_size*4, dropout=dropout)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer, encoder_norm)

    def forward(self, input_ids_list):
        emb, padding_mask = self.emb(input_ids_list)
        emb = self.encoder(emb.transpose(0,1), src_key_padding_mask=padding_mask).transpose(0,1)
        return emb   # [1, seq_len, hidden_size]

import copy
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer_Encoder2(nn.Module):
    def __init__(self, n_layer1, n_layer2, hidden_size, num_attention_heads, vocab_size, max_position_size1, max_position_size2, max_segment, word_emb_padding_idx, dropout):
        super(Transformer_Encoder2, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=hidden_size*4, dropout=dropout)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder1 = Transformer_Encoder(n_layer1, hidden_size, num_attention_heads, vocab_size, max_position_size1, max_segment, word_emb_padding_idx, dropout)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, n_layer2, encoder_norm)
        self.max_position_size2 = max_position_size2
      
    def forward(self, input_med_ids_list):
        emb_list = []
        for input_ids_list in input_med_ids_list:
            emb = self.encoder1(input_ids_list)
            emb_list.append(emb[:,0])
        if len(emb_list) < self.max_position_size2:
            emb_list += [torch.zeros_like(emb[:,0])]*(self.max_position_size2-len(emb_list))
        emb = torch.cat(emb_list, dim=0).unsqueeze(0)
        padding_mask = torch.BoolTensor([0]*len(input_med_ids_list)+[1]*(self.max_position_size2-len(input_med_ids_list))).unsqueeze(0)
        emb = self.encoder2(emb.transpose(0,1), src_key_padding_mask=padding_mask).transpose(0,1)
        return emb

# For multi-visit scenario
class Mul_Attention(nn.Module):
    def __init__(self, hidden_size, device):
        super(Mul_Attention, self).__init__()
        self.key = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.q = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        # self.key = nn.Linear(hidden_size, hidden_size)
        self.device = device

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            assert scores.shape == mask.shape
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, input_seq_rep, k_mul):
        # input: [1, seq, dim]
        shape = input_seq_rep.shape
        input_seq_key = self.key(input_seq_rep)
        input_seq_q = self.q(input_seq_rep)
        mask = torch.zeros((shape[0], shape[1], shape[1])).to(self.device)
        for i in range(shape[1]):
            for j in range(i-k_mul, i+1):
                if j>=0:
                    mask[0, i, j] = 1

        out, attn = self.attention(input_seq_q, input_seq_key, input_seq_key, mask=mask)
        return out, attn

