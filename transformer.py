# python imports
import os
import copy
import random
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# single attention head module
def attention(query, key, value, mask=None, dropout=None, verbose=False):
    
    # get dimensions
    d_k = query.size(-1)
    
    # scaled dot-product implementation
    # scores = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(d_k)
    
    # another implementation using torch.einsum
    # query  -> (n_batches, heads, query length, d_k) 
    # key    -> (n_batches, heads, key length, d_k)
    # scores -> (n_batches, heads, query_length, key_length)
    # where heads*d_k = d_model
    # einsum operand: 'bhqd,bhkd->bhqk'
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(d_k)
    
    # apply mask if needed
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9) # if mask == 0, replace with small epsilon
    
    # softmax function
    p_attn = scores.softmax(dim=-1)
    
    # dropout if needed
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    # print outputs for debugging
    if verbose:
        print(f"Input Dimension: {d_k}")
        # print(f"Scaled Q*K.T: {scores}")
        print(f"Masked Scaled Q*K.T: {scores.shape}")
        print(f"Softmax: {p_attn.shape}\n")
    
    # matrix multiplication with value
    return torch.einsum('bhqk,bhvd->bhqd', p_attn, value), p_attn

# multi-headed attention module
class MultiHeadedAttention(nn.Module):
    
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # check if d_model % h == 0
        assert(d_model % h == 0)
        # assume d_k = d_v
        self.d_k = d_model // h
        self.h = h
        # linear layers 
        self.linear_qkv = nn.Linear(d_model, d_model, bias=False) # W_q, W_k, W_v matrices
        self.linear_out = nn.Linear(d_model, d_model, bias=False) # W_o matrix
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        
        # get batch size
        n_batches = query.size(0)
    
        # shapes
        # Query: (n_batches, len_query, d_model) -> (n_batches, h, len_query, d_k)
        # Key:   (n_batches, len_key, d_model)   -> (n_batches, h, len_key, d_k)
        # Value: (n_batches, len_value, d_model) -> (n_batches, h, len_value, d_v)
        
        # pass thru linear layers to get query, key and value vectors
        query, key, value = self.linear_qkv(query), self.linear_qkv(key), self.linear_qkv(value)
        
        # reshape to pass thru attention layer
        query = query.reshape(n_batches, self.h, -1, self.d_k)
        key = key.reshape(n_batches, self.h, -1, self.d_k)
        value = value.reshape(n_batches, self.h, -1, self.d_k)

        # apply attention in batch, x -> (n_batches, heads, len_query, d_v)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) 
        
        # concat x across all the attention heads 
        x = x.transpose(1,2).contiguous().view(n_batches, -1, self.h*self.d_k)
        return self.linear_out(x)

# positional encoding module (from The Annotated Transformer)
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(00., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

# transformer (encoder) block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, dropout, d_ff):
        super(TransformerBlock, self).__init__()
        # layer definitions
        self.attention = MultiHeadedAttention(heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask):
        # 1) multihead attention layer
        attention = self.attention(query, key, value, mask)
        # 2) norm layer (with skip connection) and dropout
        x = self.norm1(query + self.dropout(attention))
        # 3) feed forward layer
        x_ff = self.feed_forward(x)
        # 4) norm layer (with skip connection) and dropout
        out = self.dropout(self.norm2(x_ff + x))
        return out
    
# full encoder module
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, max_seq_len, d_model, n_layers, heads, d_ff, dropout):
        super(Encoder, self).__init__()
        # create n_layers of transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(d_model, heads, dropout, d_ff) for _ in range(n_layers)])  
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # embedding and positional encoding layers
        self.word_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
    
    def forward(self, x, mask):
        # get input embedding and positional encoding
        x = self.positional_encoding(self.word_embedding(x))
        # go thru each Transformer layer
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x
    
# decoder block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, dropout, d_ff):
        super(DecoderBlock, self).__init__()
        # define layers
        self.attention = MultiHeadedAttention(heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.transformer_block = TransformerBlock(d_model, heads, dropout, d_ff) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key, value, src_mask, trg_mask):
        # 1) multi-headed self attention (from target)
        attention = self.attention(x, x, x, trg_mask)
        # 2) norm layer (with skip connection) + dropout
        query = self.norm1(x + self.dropout(attention))
        # 3) multi-headed attention with decoder input (query) and encoder input (key, value) 
        out = self.attention(query, key, value, src_mask)
        return out

# decoder module        
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, max_seq_len, d_model, n_layers, heads, d_ff, dropout):
        super(Decoder, self).__init__()
        # create n_layers of decoder blocks
        self.layers = nn.ModuleList([DecoderBlock(d_model, heads, dropout, d_ff) for _ in range(n_layers)])
        # embedding and positional encoding layers
        self.word_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # final fully connected layer
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        # get source and target masks
        src_mask = src_mask
        trg_mask = trg_mask
        # get input embedding and positional encoding
        x = self.positional_encoding(self.word_embedding(x)) 
        for layer in self.layers:
            # DecoderBlock forward arguments = (x, key, value, src_mask, trg_mask)
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        return self.fc_out(x)
    
# full Transformer model
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 d_model=256,
                 n_layers=8,
                 d_ff=512,
                 heads=8,
                 dropout=0.0,
                 max_seq_len=100
                 ):
        super(Transformer, self).__init__()
        # define encoder/decoder layers
        self.encoder = Encoder(src_vocab_size, max_seq_len, d_model, n_layers, heads, d_ff, dropout)
        self.decoder = Decoder(trg_vocab_size, max_seq_len, d_model, n_layers, heads, d_ff, dropout)
        # padding indices
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        
    def make_src_mask(self, src):
        # src mask shape -> (n_batches, 1, 1, seq_len)
        return (src != self.src_pad_idx) .unsqueeze(1).unsqueeze(2)
    
    def make_trg_mask(self, trg):
        # trg mask shape -> (n_batches, 1, seq_len, seq_len)
        n_batches, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(n_batches, 1, trg_len, trg_len)
        return trg_mask
        
    def forward(self, src, trg):
        # get masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # encoder and decoders
        src_out = self.encoder(src, src_mask)
        out = self.decoder(trg, src_out, src_mask, trg_mask)
        return out 


def main():
    # set deterministic behavior
    seed = 42
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # input and output sizes
    src_vocab_size = 8
    trg_vocab_size = 12
    n_batches = 2
    src_seq_len = 4
    trg_seq_len = 7

    # create source and target tensors
    src_pad_idx = 0
    trg_pad_idx = 0
    src = torch.randint(0, src_vocab_size, (n_batches, src_seq_len))
    trg = torch.randint(0, trg_vocab_size, (n_batches, trg_seq_len))

    # create Transformer object and pass thru source and target
    my_transformer = Transformer(src_vocab_size,
                                 trg_vocab_size,
                                 src_pad_idx,
                                 trg_pad_idx,
                                 d_model=256,
                                 n_layers=8,
                                 d_ff=512,
                                 heads=8,
                                 dropout=0.0,
                                 max_seq_len=100)
    out = my_transformer(src, trg[:,:-1])
    
    # check output dimensions
    print(f"Expected Output Dimensions: ({n_batches}, {trg_seq_len-1}, {trg_vocab_size})")
    print(f"Output Shape: {out.shape}")

if __name__ == "__main__":
    main()
    
