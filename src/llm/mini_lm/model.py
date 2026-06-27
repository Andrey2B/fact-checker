"""
Минимальная языковая модель для верификации фактов.
Архитектура: Transformer Decoder (по книге Raschka "Build LLM from Scratch")

Задача: claim + evidence → verdict (3 класса)
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention с causal mask
        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Causal mask
        mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff   = FeedForward(d_model, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual + pre-norm
        x = x + self.ff(self.ln2(x))
        return x


class MiniVerifier(nn.Module):
    """
    Маленький трансформер для верификации фактов.

    Вход:  токенизированный текст [claim] + [SEP] + [evidence]
    Выход: логиты для 3 классов:
           0 = SUPPORTED
           1 = REFUTED
           2 = NOT_ENOUGH_INFO
    """

    # Конфигурация (маленькая модель ~4M параметров)
    DEFAULT_CONFIG = {
        "vocab_size":  8192,
        "d_model":     256,
        "n_heads":     8,
        "n_layers":    4,
        "max_seq_len": 512,
        "n_classes":   3,
        "dropout":     0.1,
    }

    LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        self.max_seq_len = cfg["max_seq_len"]
        self.d_model     = cfg["d_model"]

        # Embeddings
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.pos_emb   = nn.Embedding(cfg["max_seq_len"], cfg["d_model"])
        self.drop_emb  = nn.Dropout(cfg["dropout"])

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["dropout"])
            for _ in range(cfg["n_layers"])
        ])

        self.ln_f = nn.LayerNorm(cfg["d_model"])

        # Классификационная голова
        self.classifier = nn.Sequential(
            nn.Linear(cfg["d_model"], cfg["d_model"] // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["d_model"] // 2, cfg["n_classes"]),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape

        # Token + positional embeddings
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # представление последнего не-padding токена для классификации
        if attention_mask is not None:
            # последний реальный токен
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            idx     = lengths.clamp(max=T - 1)
            pooled  = x[torch.arange(B), idx]        # (B, d_model)
        else:
            pooled = x[:, -1, :]                     # (B, d_model)

        return self.classifier(pooled)               # (B, n_classes)

    def predict(self, input_ids: torch.Tensor) -> list[str]:
        """Предсказать класс для батча."""
        with torch.no_grad():
            logits = self.forward(input_ids)
            preds  = logits.argmax(dim=-1)
        return [self.LABELS[p.item()] for p in preds]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)