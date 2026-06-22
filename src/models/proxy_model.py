"""
Proxy Working Memory Model for pre-training.

Wraps the existing model architecture (WorkingMemoryModel or
AttentionWorkingMemoryModel) and replaces the 3-class classifier
with multi-head proxy heads for feature recall pre-training.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .perceptual import PerceptualModule
from .cognitive import CognitiveModule
from .proxy_heads import ProxyHeads
from .attention import FeatureChannelAttention


class ProxyWorkingMemoryModel(nn.Module):
    def __init__(self, perceptual: PerceptualModule, cognitive: CognitiveModule,
                 hidden_size: int, num_identities: int,
                 attention: Optional[FeatureChannelAttention] = None,
                 num_locations: int = 4, num_categories: int = 4):
        super().__init__()
        self.perceptual = perceptual
        self.cognitive = cognitive
        self.hidden_size = hidden_size
        self.attention = attention
        self.is_attention = attention is not None

        self.proxy_heads = ProxyHeads(
            hidden_size=hidden_size,
            num_identities=num_identities,
            num_locations=num_locations,
            num_categories=num_categories,
        )

    def forward(self, images: torch.Tensor, task_vector: torch.Tensor,
                return_cnn_activations: bool = False):
        B, T = images.shape[0], images.shape[1]
        task_dim = task_vector.shape[-1]

        x = images.reshape(B * T, *images.shape[2:])

        if return_cnn_activations:
            emb, feat_map = self.perceptual(x, return_feature_map=True)
            cnn_activations = feat_map.mean(dim=[2, 3])
            cnn_activations = cnn_activations.view(B, T, self.hidden_size)
        else:
            emb, _ = self.perceptual(x)
            cnn_activations = None

        emb = emb.view(B, T, self.hidden_size)

        if self.is_attention:
            gated_features, gates = self.attention(emb, task_vector, return_gates=False)
            features = gated_features
        else:
            features = emb

        task_rep = task_vector.unsqueeze(1).expand(B, T, task_dim)
        cog_in = torch.cat([features, task_rep], dim=-1)

        outputs, final_state, hidden_seq = self.cognitive(cog_in)

        if return_cnn_activations:
            return hidden_seq, final_state, cnn_activations
        return hidden_seq, final_state

    def get_proxy_logits(self, hidden_seq: torch.Tensor,
                         task_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.proxy_heads(hidden_seq, task_vector)

    def predict_for_task(self, hidden_seq: torch.Tensor,
                         task_feature: str) -> torch.Tensor:
        head = self.proxy_heads.get_head_for_feature(task_feature)
        return head(hidden_seq)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, task_vector: torch.Tensor,
                task_feature: str = "location"):
        hidden_seq, _ = self.forward(images, task_vector)
        logits = self.predict_for_task(hidden_seq, task_feature)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs, hidden_seq
