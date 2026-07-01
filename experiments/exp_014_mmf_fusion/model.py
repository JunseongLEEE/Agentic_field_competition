"""exp_014 Multi-Modal Fusion Net.

Three towers fused end-to-end:
  1. text  : multilingual encoder (mean-pooled) -> 256
  2. seq   : history-trajectory TransformerEncoder -> 128
  3. tab   : categorical embeddings + normalized numerics MLP -> 128
Fusion: concat(512) -> LayerNorm -> MLP(512->256, GELU, dropout) -> 14 logits.

Defined self-contained so script.py can import it from the packaged model/ dir.
"""
import math

import torch
import torch.nn as nn


class SeqTower(nn.Module):
    def __init__(self, n_step_vocab, max_seq_len, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        d_step, d_role, d_fail = 96, 16, 16
        assert d_step + d_role + d_fail == d_model
        self.step_emb = nn.Embedding(n_step_vocab, d_step, padding_idx=0)
        self.role_emb = nn.Embedding(4, d_role)
        self.fail_emb = nn.Embedding(3, d_fail)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.d_model = d_model

    def forward(self, toks, roles, fails, mask):
        # mask: 1 for valid, 0 for pad
        B, L = toks.shape
        pos = torch.arange(L, device=toks.device).unsqueeze(0).expand(B, L)
        x = torch.cat(
            [self.step_emb(toks), self.role_emb(roles), self.fail_emb(fails)], dim=-1
        )
        x = x + self.pos_emb(pos)
        pad_mask = mask == 0  # True where pad
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return pooled


class TabTower(nn.Module):
    def __init__(self, cat_cards, n_num, out_dim=128):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_total = 0
        for c in cat_cards:
            d = int(min(24, max(2, round(1.6 * (c ** 0.5)))))
            self.embs.append(nn.Embedding(c, d))
            emb_total += d
        self.n_num = n_num
        in_dim = emb_total + n_num
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim), nn.GELU(),
        )

    def forward(self, cats, nums):
        parts = [emb(cats[:, i]) for i, emb in enumerate(self.embs)]
        parts.append(nums)
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


class MMFNet(nn.Module):
    def __init__(self, encoder, hidden_size, cat_cards, n_num,
                 n_step_vocab, max_seq_len, n_classes=14,
                 d_text=256, d_seq=128, d_tab=128, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.text_proj = nn.Sequential(nn.Linear(hidden_size, d_text), nn.GELU())
        self.seq_tower = SeqTower(n_step_vocab, max_seq_len, d_model=d_seq)
        self.tab_tower = TabTower(cat_cards, n_num, out_dim=d_tab)
        fuse_in = d_text + d_seq + d_tab
        self.norm = nn.LayerNorm(fuse_in)
        self.head = nn.Sequential(
            nn.Linear(fuse_in, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def text_vec(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.text_proj(pooled)

    def forward(self, input_ids, attention_mask,
                seq_toks, seq_roles, seq_fails, seq_mask, cats, nums):
        t = self.text_vec(input_ids, attention_mask)
        s = self.seq_tower(seq_toks, seq_roles, seq_fails, seq_mask)
        b = self.tab_tower(cats, nums)
        z = torch.cat([t, s, b], dim=-1)
        z = self.norm(z)
        return self.head(z)
