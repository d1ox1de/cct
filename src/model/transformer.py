import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .stochastic_depth import DropPath


class Attention(nn.Module):
# https://arxiv.org/pdf/2104.05704.pdf

    def __init__(self,
                embed_dim: int,
                num_heads: int = 8,
                attention_dropout: float = 0.1,
                projection_dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by the number of heads."
        self.num_heads = num_heads
        head_dim = embed_dim // self.num_heads
        self.scale = head_dim ** (-0.5)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(projection_dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        num_batches, sequence_len, num_channels = x.shape # for patching it would be sequence_len = path_size*path_size + 1 (class_token)
        qkv = self.qkv(x).reshape(num_batches, sequence_len, 3, self.num_heads, num_channels // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv ~ (3, num_batches, self.num_heads, sequence_len, num_channels // self.num_heads( -> new_embed_dim))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale # attn ~ (num_batches, self.num_heads, sequence_len)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn @ v ~ (num_batches, self.num_heads, sequence_len, new_embed_dim)
        x = (attn @ v).transpose(1, 2).reshape(num_batches, sequence_len, num_channels)
        # x ~ (num_batches, sequence_len, num_channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                dim_model: int,
                num_heads: int,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                attention_dropout: float = 0.1,
                drop_path_rate: float = 0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(dim_model)
        self.self_attn = Attention(
            embed_dim=dim_model, num_heads=num_heads,
            attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu


    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # src ~ (num_batches, sequence_len, num_channels)
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))

        src = src + self.drop_path(self.dropout2(src2))

        return src


class TransformerClassifier(nn.Module):

    def __init__(self,
                num_classes: int,
                embed_dims: int = 768,
                num_layers: int = 12,
                num_heads: int = 12,
                fc_ration: float = 4.,
                dropout: float = 0.1,
                attention_dropout: float = 0.1,
                stochastic_depth: float = 0.1,
                positional_embed: str = 'learnable',
                sequence_pool: bool = True,
                sequence_length: Optional[int] = None):
        super().__init__()
        positional_embed = positional_embed if positional_embed in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embed_dims * fc_ration)
        
        self.embed_dims = embed_dims
        self.sequence_length = sequence_length
        self.sequence_pool = sequence_pool
        self.num_tokens = 0
        
        assert sequence_length is not None or positional_embed == 'none', \
            f"Positional embedding is set to {positional_embed} and" \
            f" the sequence length was not specified."
        
        if not sequence_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dims), requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = nn.Linear(self.embed_dims, 1)
        
        if positional_embed != 'none':
            if positional_embed == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embed_dims), requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_embed = nn.Parameter(self.sinusoidal_embed(sequence_length, embed_dims), requires_grad=False)
        else:
            self.positional_emb = None
        
        self.dropout = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                dim_model=embed_dims, num_heads=num_heads,
                dim_feedforward=dim_feedforward, dropout=dropout,
                attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dims)
        self.fc = nn.Linear(embed_dims, num_classes)
        
        self.apply(self.init_weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0,  self.n_channels - x.size(1)), mode='constant', value=0)
        if not self.sequence_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1) # -1 means not changing the size of that dimension
            x = torch.cat((cls_token, x), dim=1)
        if self.positional_emb is not None:
            x += self.positional_emb
        
        x = self.dropout(x)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        if self.sequence_pool:
            # pool over the entire sequence of data since it contains relevant information 
            # (B, seq_len, embed_dims) -> (B, embed_dims)
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x


    @staticmethod
    def sinusoidal_embed(seq_len: int, dims: int) -> torch.Tensor:
        pos_embed = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dims)) for i in range(dims)]
                                        for p in range(seq_len)])
        pos_embed[:, 0::2] = torch.sin(pos_embed[:, 0::2])
        pos_embed[:, 1::2] = torch.cos(pos_embed[:, 1::2])
        return pos_embed.unsqueeze(0)


    @staticmethod
    def init_weights(submodule):
        if isinstance(submodule, nn.Linear):
            nn.init.trunc_normal_(submodule.weight, std=0.02)
            if isinstance(submodule, nn.Linear) and submodule.bias is not None:
                nn.init.constant_(submodule.bias, 0.)
        elif isinstance(submodule, nn.LayerNorm):
            nn.init.constant_(submodule.bias, 0.)
            nn.init.constant_(submodule.weight, 1.)
