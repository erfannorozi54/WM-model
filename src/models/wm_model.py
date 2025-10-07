import torch
import torch.nn as nn
from typing import Tuple, Optional

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

    def __init__(self, perceptual: PerceptualModule, cognitive: CognitiveModule, hidden_size: int):
        super().__init__()
        self.perceptual = perceptual
        self.cognitive = cognitive
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, images: torch.Tensor, task_vector: torch.Tensor):
        B, T = images.shape[0], images.shape[1]
        # Flatten time into batch for perceptual encoder
        x = images.reshape(B * T, *images.shape[2:])  # (B*T, 3, H, W)
        emb, _ = self.perceptual(x)  # (B*T, H)
        emb = emb.view(B, T, self.hidden_size)  # (B, T, H)

        # Expand task vector across time and concatenate
        task_rep = task_vector.unsqueeze(1).expand(B, T, 3)  # (B, T, 3)
        cog_in = torch.cat([emb, task_rep], dim=-1)  # (B, T, H+3)

        outputs, final_state, hidden_seq = self.cognitive(cog_in)
        logits = self.classifier(outputs)  # (B, T, 3)
        return logits, hidden_seq, final_state

    @torch.no_grad()
    def predict(self, images: torch.Tensor, task_vector: torch.Tensor):
        logits, hidden_seq, _ = self.forward(images, task_vector)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)  # (B, T)
        return preds, probs, hidden_seq
