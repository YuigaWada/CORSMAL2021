import math
from typing import Callable, List, Optional

import torch
from torch import nn

from task1and2.corsmal_challenge.models.activation import SquaredReLU


class MultiheadedSelfAttention(nn.Module):
    """(batches, sequence_len, embed_dim) -> same size tensor"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        dropout: float = 0.05,
    ):
        super(MultiheadedSelfAttention, self).__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dim % num heads != 0"
        self.head_dim: int = embed_dim // num_heads
        self.scale: float = self.head_dim ** -0.5

        self.qkv: Callable[..., torch.Tensor] = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout: Callable[..., torch.Tensor] = nn.Dropout(dropout)
        self.projection: Callable[..., torch.Tensor] = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout: Callable[..., torch.Tensor] = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """forward

        Args:
            inputs (torch.Tensor): (batches, sequence_len, embed_dim)
            mask (torch.Tensor): (batches, sequence_len)

        Returns:
            torch.Tensor: (batches, sequence_len, embed_dim)
        """
        batches, sequence_len, _ = inputs.shape
        qkv: torch.Tensor = (
            self.qkv(inputs).reshape(batches, sequence_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv  # batches, num_heads, sequence_len, head_dim

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            qk = qk + mask.float() * -torch.finfo(inputs.dtype).max
        attn = qk.softmax(dim=-1)
        x = torch.matmul(attn, v)

        x = x.permute(0, 2, 1, 3).reshape(batches, sequence_len, self.head_dim * self.num_heads)
        x = self.attn_dropout(x)
        proj = self.projection(x)
        proj = self.proj_dropout(proj)

        return proj


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 8000,
        freq: float = 16000.0,
    ):
        super(PositionalEncoding, self).__init__()
        pos_e = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(freq) / embed_dim))
        pos_e[:, 0::2] = torch.sin(position * div)
        pos_e[:, 1::2] = torch.cos(position * div)
        pos_e = pos_e.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_e", pos_e)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs + self.pos_e[: inputs.size(0), :]  # type: ignore
        return x


class FFN(nn.Module):
    def __init__(self, embed_dim: int, expansion: int = 4, dropout: float = 0.05):
        super(FFN, self).__init__()
        self.embed_dim: int = embed_dim
        self.expansion: int = expansion
        self.fc1: Callable[..., torch.Tensor] = nn.Linear(embed_dim, embed_dim * expansion)
        self.squared_relu = SquaredReLU()
        self.fc2: Callable[..., torch.Tensor] = nn.Linear(embed_dim * expansion, embed_dim)
        self.dropout: Callable[..., torch.Tensor] = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fc1(inputs)
        x = self.squared_relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.05,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mhla = MultiheadedSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FFN(embed_dim, expansion=2)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(inputs)
        x = self.mhla(x, mask)
        internal = x + inputs
        x = self.norm(internal)
        x = self.ffn(x)
        out = x + internal
        return out


class CLSTokenAdder(nn.Module):
    def __init__(self, embed_dim: int):
        super(CLSTokenAdder, self).__init__()
        self.embed_dim = embed_dim
        self.gen_cls_token = nn.Linear(in_features=1, out_features=self.embed_dim, bias=False)
        self.ones = nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batches = inputs.shape[0]
        cls_token = self.gen_cls_token(self.ones.expand(batches, 1, 1))
        return torch.cat((cls_token, inputs), dim=1)


class TransformerEncoder(nn.Module):
    """Transformer encoder. (batches, sequence_len, embed_dim) -> same size tensor"""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.05,
    ):
        super(TransformerEncoder, self).__init__()
        self.num_layers: int = num_layers
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.cls_token_adder = CLSTokenAdder(self.embed_dim)
        self.pe = PositionalEncoding(embed_dim)
        self.first_block = TransformerEncoderBlock(
            embed_dim,
            num_heads,
            dropout,
        )
        self.layer_stack = self._make_encoder_block_stack(
            num_layers - 1,
            embed_dim,
            num_heads,
            dropout,
        )

    def _make_encoder_block_stack(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.05,
    ):
        layer_stack: List[TransformerEncoderBlock] = []

        for _ in range(num_layers):
            layer_stack.append(TransformerEncoderBlock(embed_dim, num_heads, dropout))

        return nn.Sequential(*layer_stack)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # add class token at the head of given sequence
        x = self.cls_token_adder(inputs)
        x = self.pe(x)
        x = self.first_block(x, mask)
        x = self.layer_stack(x)
        return x
