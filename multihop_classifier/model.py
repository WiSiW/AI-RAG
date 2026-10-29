import torch
from torch import nn
from transformers import AutoModel


class BertWithFeatures(nn.Module):
    def __init__(self, model_name: str, feature_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + 16, 2),
        )

    def forward(self, input_ids, attention_mask, features):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0]
        fused = torch.cat([cls, self.feature_layer(features)], dim=-1)
        return self.classifier(fused)
