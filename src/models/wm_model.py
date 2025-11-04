import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from .perceptual import PerceptualModule
from .cognitive import CognitiveModule


class WorkingMemoryModel(nn.Module):
    """
    Two-stage sensory-cognitive model for N-back WM tasks.

    Components:
    - PerceptualModule (ResNet50-based) -> embedding of size H
    - CognitiveModule (RNN/GRU/LSTM) taking [embedding ; task_one_hot] per timestep
    - Classifier projecting hidden state to 3 response classes

    Forward input:
        images: (B, T, 3, H, W)
        task_vector: (B, 3) one-hot
    Returns:
        logits: (B, T, 3)
        hidden_seq: (B, T, H)
        final_state: RNN final state (module-dependent)
    """

    def __init__(self, perceptual: PerceptualModule, cognitive: CognitiveModule, hidden_size: int, classifier_layers: Optional[List[int]] = None):
        super().__init__()
        self.perceptual = perceptual
        self.cognitive = cognitive
        self.hidden_size = hidden_size
        if classifier_layers:
            layers = []
            in_dim = hidden_size
            for dim in classifier_layers:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(nn.ReLU(inplace=True))
                in_dim = dim
            layers.append(nn.Linear(in_dim, 3))
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, images: torch.Tensor, task_vector: torch.Tensor, return_cnn_activations: bool = False):
        B, T = images.shape[0], images.shape[1]
        # Flatten time into batch for perceptual encoder
        x = images.reshape(B * T, *images.shape[2:])  # (B*T, 3, H, W)
        
        # Optionally capture CNN penultimate layer activations
        if return_cnn_activations:
            emb, feat_map = self.perceptual(x, return_feature_map=True)  # (B*T, H), (B*T, H, H', W')
            # Global average pool the feature map for analysis
            cnn_activations = feat_map.mean(dim=[2, 3])  # (B*T, H)
            cnn_activations = cnn_activations.view(B, T, self.hidden_size)  # (B, T, H)
        else:
            emb, _ = self.perceptual(x)  # (B*T, H)
            cnn_activations = None
        
        emb = emb.view(B, T, self.hidden_size)  # (B, T, H)

        # Expand task vector across time and concatenate
        task_rep = task_vector.unsqueeze(1).expand(B, T, 3)  # (B, T, 3)
        cog_in = torch.cat([emb, task_rep], dim=-1)  # (B, T, H+3)

        outputs, final_state, hidden_seq = self.cognitive(cog_in)
        logits = self.classifier(outputs)  # (B, T, 3)
        
        if return_cnn_activations:
            return logits, hidden_seq, final_state, cnn_activations
        return logits, hidden_seq, final_state

    @torch.no_grad()
    def predict(self, images: torch.Tensor, task_vector: torch.Tensor):
        logits, hidden_seq, _ = self.forward(images, task_vector)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)  # (B, T)
        return preds, probs, hidden_seq
