# my-transformer

This repo is still being developed, so please come check it out in the future!

<!-- ABOUT THE PROJECT -->
## About my-transformer

Welcome to my-transformer! This repo contains my implementation of a transformer as laid out in the seminal paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) in PyTorch. In addition to the paper, I used several other resources to implement it myself. They are cited in the Resources section below. 

I created this implementation primarily for educational purposes. The main goals are the following:

* Learn and understand how transformer architecture works 
* Implement one myself from scratch
* Help others understand how transformers work

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To run my implementation of transformer architecture, please follow the instructions below.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ryanbae89/my-transformer.git
   ```
2. Install required Python libraries
   ```sh
   conda env create -f environment.yml
   ```
3. Run the transformer script
   ```sh
   conda activate 
   python transformer.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- TRANSFORMER ARCHITECTURE -->
## Transformer Architecture

Transformers have revolutionized 

The transformer implemented in the Attention is All You Need paper is called _Autoregressive Transformer_. This is because the decoder block in the transformer is autoregressive, meaning it is only able to "see" the data upto the current token being inferenced on. 

Let's go over in detail each part of a transformer architecture with relevant code. 

### Overall Architecture

The full transformer architecture is often illustrated using the following visualization from the Attention is All You Need Paper.

[<img src="https://github.com/ryanbae89/ryanbae89.github.io/blob/master/images/transformer-architecture.PNG?raw=true" width="700">]

The vanilla transformer architecture is fairly straightforward, compared to some other more complex models. It consists of 3 main parts:

1. Encoder
2. Decoder
3. Output


The full transformer model in Pytorch implementation looks like the following:

```python
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
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_trg_mask(self, trg):
        # trg mask shape -> (n_batches, 1, seq_len, seq_len)
        if trg is None:
            return None
        else:
            n_batches, trg_len = trg.shape
            trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(n_batches, 1, trg_len, trg_len)
            return trg_mask
        
    def forward(self, src, trg=None):
        # get masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # encoder and decoders
        src_out = self.encoder(src, src_mask)
        out = self.decoder(trg, src_out, src_mask, trg_mask)
        
        # generation mode (no target)
        if trg is None:
            loss = None
        # training mode
        else:
            # reshape logits and targets
            n_batches, seq_len, embed_size = out.shape
            out = out.view(n_batches*seq_len, embed_size) 
            trg = trg.contiguous().view(n_batches*seq_len)
            loss = F.cross_entropy(out, trg)

        return out, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (batch_size, seq_len) array of indicies in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # get the last step in the sequence dimension
            logits = logits[:, -1, :] # becomes (batch_size, embed_size)
            # convert to probabilities
            probs = F.softmax(logits, dim=1) # (batch_size, embed_size)
            # get token index of prediction by sampling from probs
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (batch_size, seq_len + 1)
        return idx
```

Let's go over the input parameters to the model one by one. 

```
src_vocab_size  = Size of the source text vocabulary. 
trg_vocab_size  = This is the size of the target text vocabulary.
src_pad_idx     = This is the index of the padding token for the source text.
trg_pad_idx     = This is the index of the padding token for the target text.
d_model         = Number of dimensions in each attention head. 
n_layers        = Number of layers in the stack of the encoder and decoder.
d_ff            = Number of dimensions in the feed forward layers.
heads           = Number of attention heads in multihead attention layer.
dropout         = Fraction of parameters to zero in dropout layers.
max_seq_len     = Max length of source/target sequences.
```

### Scaled Dot-Product Attention

Let's start by diving deeper into the attention layer, which is at the heart of the transformer architecture. 

```python
# single attention head module
def attention(query, key, value, mask=None, dropout=None, verbose=False):
    
    # get dimensions
    d_k = query.size(-1)
    
    # vector embeddings
    # query  -> (n_batches, heads, query length, d_k) 
    # key    -> (n_batches, heads, key length, d_k)
    # scores -> (n_batches, heads, query_length, key_length)
    # where heads*d_k = d_model
    
    # scaled dot-product implementation
    # scores = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(d_k)
    
    # another implementation using torch.einsum
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
    
    # matrix multiplication with value
    return torch.einsum('bhqk,bhvd->bhqd', p_attn, value), p_attn
```

### Multi-Headed Attention

```python
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
```

### Positional Encoding

```python
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
```

### Encoder

```python
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
```


```python
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
```

### Decoder

```python
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
```


```python
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
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Ryan Bae - [@twitter_handle](https://twitter.com/twitter_handle) - ryanbae89@gmail.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REFERENCES -->
## References

Below are the references I used to build this repo and create my own working implementation of transformer. Their previous work in this area have been tremendous in helping me create my own implementation.

* [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [Let's build GPT: from scratch, in code, spelled out., Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5706s)
* [Pytorch Transformers from Scratch (Attention is all you need), Aladdin Persson](https://www.youtube.com/watch?v=U0s0f995w14)
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>