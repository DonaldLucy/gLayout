from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FTTransformerConfig:
    num_numeric_features: int
    categorical_cardinalities: list[int]
    output_dim: int
    d_token: int = 192
    n_blocks: int = 6
    attention_n_heads: int = 8
    attention_dropout: float = 0.10
    ffn_dropout: float = 0.15
    residual_dropout: float = 0.05
    token_dropout: float = 0.05
    ffn_hidden_multiplier: float = 4.0


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias = nn.Parameter(torch.empty(num_features, d_token))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, cardinalities: list[int], d_token: int) -> None:
        super().__init__()
        offsets = [0]
        for cardinality in cardinalities[:-1]:
            offsets.append(offsets[-1] + cardinality)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.embedding = nn.Embedding(sum(cardinalities), d_token)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x + self.offsets.unsqueeze(0))


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        return left * F.relu(right)


class FeedForward(nn.Module):
    def __init__(self, d_token: int, multiplier: float, dropout: float) -> None:
        super().__init__()
        d_hidden = int(d_token * multiplier)
        self.net = nn.Sequential(
            nn.Linear(d_token, d_hidden * 2),
            ReGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_token),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        ffn_hidden_multiplier: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(d_token)
        self.ff = FeedForward(d_token=d_token, multiplier=ffn_hidden_multiplier, dropout=ffn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.residual_dropout(attn_output)
        ff_output = self.ff(self.ff_norm(x))
        x = x + self.residual_dropout(ff_output)
        return x


class FTTransformer(nn.Module):
    def __init__(self, config: FTTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.num_tokenizer = (
            NumericalFeatureTokenizer(config.num_numeric_features, config.d_token)
            if config.num_numeric_features > 0
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(config.categorical_cardinalities, config.d_token)
            if config.categorical_cardinalities
            else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_token))
        self.token_dropout = nn.Dropout(config.token_dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=config.d_token,
                    n_heads=config.attention_n_heads,
                    attention_dropout=config.attention_dropout,
                    ffn_dropout=config.ffn_dropout,
                    residual_dropout=config.residual_dropout,
                    ffn_hidden_multiplier=config.ffn_hidden_multiplier,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_token),
            nn.Linear(config.d_token, config.output_dim),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        batch_size = x_num.shape[0] if x_num.numel() else x_cat.shape[0]
        if self.num_tokenizer is not None:
            tokens.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            tokens.append(self.cat_tokenizer(x_cat))
        if not tokens:
            raise RuntimeError("FTTransformer received no numeric or categorical features.")
        token_tensor = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, token_tensor], dim=1)
        x = self.token_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
