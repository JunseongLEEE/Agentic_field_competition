"""ActionClassifier + Focal CE for exp_015."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ActionClassifier(nn.Module):
    def __init__(self, model_name_or_path, n_classes=14, dropout=0.1, local_files_only=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)
        self.n_classes = n_classes

    def forward(self, input_ids, attention_mask, log_prior=None, tau=1.0):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        if self.training and log_prior is not None:
            logits = logits - tau * log_prior
        return logits


def focal_ce(logits, labels, class_weights, gamma=2.0):
    ce = F.cross_entropy(logits, labels, reduction="none", weight=class_weights)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()
