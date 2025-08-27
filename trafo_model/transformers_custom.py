import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        # Learnable positional embeddings stored as a parameter
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)

        # Add positional embedding for the current sequence length
        return x + self.pos_embedding[:, :seq_len, :]


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, hidden_dimension, num_attention_heads, dropout):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert hidden_dimension % num_attention_heads == 0
        self.positional_encoding = LearnablePositionalEncoding(
            d_model=512, max_len=1024
        )
        self.hidden_dimension = hidden_dimension
        self.num_attention_heads = num_attention_heads
        self.head_dimension = hidden_dimension // num_attention_heads

        self.W_q = nn.Linear(hidden_dimension, hidden_dimension)
        self.W_k = nn.Linear(hidden_dimension, hidden_dimension)
        self.W_v = nn.Linear(hidden_dimension, hidden_dimension)

        self.W_o = nn.Linear(hidden_dimension, hidden_dimension)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dimension])).to(device)

    def split_heads(self, item, batch_size):
        item = item.view(batch_size, -1, self.num_attention_heads, self.head_dimension)
        item = item.permute(0, 2, 1, 3)
        return item

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]
        # Step 3 from the image
        # ==========================================================#
        # ==========================================================#

        Q = self.W_q(query)  # Q,query shape is (bsiz,qlen,hdim)
        K = self.W_k(key)  # K,key shape is (bsiz,klen,hdim)
        V = self.W_v(value)  # V,value shape is (bsiz,vlen,hdim)
        Q = Q + self.positional_encoding(Q)
        K = K + self.positional_encoding(K)
        Q = self.split_heads(Q, batch_size)  # Q shape(bsiz,n_attn_heads,qlen,head_dim)
        K = self.split_heads(K, batch_size)  # K shape(bsiz,n_attn_heads,klen,head_dim)
        V = self.split_heads(V, batch_size)  # V shape(bsiz,n_attn_heads,vlen,head_dim)

        # ==========================================================#
        # ==========================================================#

        # Step 4 from the image
        # ==========================================================#
        # ==========================================================#

        # permute because Q = (bsize,n_attn_heads,qlen,head_dim)
        #             and K = (bsize,n_attn_heads,klen,head_dim)
        # we need to multiply across batches and corresponding attention heads,
        # while following matrix multiplication compatibility
        # Q(bsize,n_attn_heads,qlen,hdim) x K(bsize,n_attn_heads,hdim,klen)
        # mainly we have to ensure that m1 = (qlen,hdim)  x m2 = (hdim,klen)
        #                       gives  m3   = (qlen,klen)

        # Q x K.T / sqrt(d_k)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = (bsize,n_attn_heads,qlen,klen)

        # the below masking is done so that after softmax,useless and very low values go to 0.
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e15)

        # softmax is applied over last dimension, across all batches, across all heads
        # softmax(Q x K.T / sqrt(d_k))
        attention = torch.softmax(energy, dim=-1)

        # apply attention_score x value
        # attention = (bsiz,n_attn_heads,qlen,klen)
        # V shape(bsiz,n_attn_heads,vlen,hdim)

        # softmax(Q x K.T / sqrt(d_k)) * V
        # equivalent to attention_scores * V

        attention_scored_value = torch.matmul(self.dropout(attention), V)
        # as klen = vlen, attention_scored_value(bsiz,n_attn_heads,qlen,hdim)

        attention_scored_value = attention_scored_value.permute(0, 2, 1, 3).contiguous()

        # attention_scored_value(bsiz,qlen,n_attn_heads,hdim)
        # contiguous makes a copy of the tensor such that the order of its elements in memory
        # is the same as if it had been created from scratch with the same data.
        # ==========================================================#
        # ==========================================================#

        # Step 5 from the image
        # ==========================================================#
        # ==========================================================#

        attention_scored_value = attention_scored_value.view(
            batch_size, -1, self.hidden_dimension
        )
        # attention_scored_value = (bsiz,qlen,h_dim)

        attention_contexts_Z = self.W_o(attention_scored_value)
        # ==========================================================#
        # ==========================================================#

        return attention_contexts_Z, attention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, _ = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(
            d_model, d_model
        )  # You may replace this depending on output task
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: Tensor of shape (batch_size, src_seq_len, d_model)
        tgt: Tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)

        output = self.fc_out(dec_output)
        return output
