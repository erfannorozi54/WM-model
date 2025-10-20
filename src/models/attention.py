"""
Task-Guided Attention Module for Working Memory Model.

This module implements a task-guided attention mechanism that allows the model
to focus on task-relevant features in the visual feature map. The attention
is conditioned on the task identity vector.

Key Components:
- TaskGuidedAttention: Attention mechanism that uses task vector to weight spatial features
- AttentionWorkingMemoryModel: Full model with attention between perceptual and cognitive modules

The attention mechanism computes spatial attention weights based on:
1. Visual feature map from CNN (B, C, H, W)
2. Task identity vector (B, 3)

Output: Task-focused context vector (B, C) that emphasizes task-relevant spatial locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .perceptual import PerceptualModule
from .cognitive import CognitiveModule


class TaskGuidedAttention(nn.Module):
    """
    Task-Guided Spatial Attention Module.
    
    This module computes spatial attention weights over the CNN feature map,
    conditioned on the task identity. The attention allows the model to focus
    on task-relevant spatial locations.
    
    Architecture:
    1. Project task vector to attention query space
    2. Project spatial features to attention key space
    3. Compute attention scores via dot product
    4. Apply softmax to get attention weights
    5. Weighted sum of spatial features
    
    Args:
        feature_dim: Dimension of CNN features (C)
        task_dim: Dimension of task vector (3 for location/identity/category)
        hidden_dim: Hidden dimension for attention computation
        dropout: Dropout rate for attention weights
    """
    
    def __init__(
        self,
        feature_dim: int,
        task_dim: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim or feature_dim
        
        # Task encoder: project task vector to query space
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Spatial feature encoder: project features to key space
        # Uses 1x1 conv to maintain spatial structure
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        
        # Attention scoring: compute compatibility between query and keys
        self.attention_score = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        
        # Optional: learnable scaling factor
        self.scale = nn.Parameter(torch.tensor(1.0 / (self.hidden_dim ** 0.5)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        feature_map: torch.Tensor,
        task_vector: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute task-guided attention over spatial features.
        
        Args:
            feature_map: CNN feature map (B, C, H, W)
            task_vector: Task identity one-hot (B, 3)
            return_attention_weights: If True, return attention weights for visualization
        
        Returns:
            context: Attention-weighted feature vector (B, C)
            attention_weights: Spatial attention weights (B, 1, H, W) if return_attention_weights=True
        """
        B, C, H, W = feature_map.shape
        
        # Encode task vector to query (B, hidden_dim)
        task_query = self.task_encoder(task_vector)  # (B, hidden_dim)
        
        # Encode spatial features to keys (B, hidden_dim, H, W)
        feature_keys = self.feature_encoder(feature_map)  # (B, hidden_dim, H, W)
        
        # Compute attention scores via broadcasting
        # Expand task_query to match spatial dimensions
        task_query_expanded = task_query.view(B, self.hidden_dim, 1, 1)  # (B, hidden_dim, 1, 1)
        
        # Element-wise product + reduction (can also use additive attention)
        # Here we use multiplicative attention followed by projection
        attended_features = feature_keys * task_query_expanded  # (B, hidden_dim, H, W)
        
        # Compute attention scores (B, 1, H, W)
        attention_scores = self.attention_score(attended_features) * self.scale
        
        # Normalize with softmax over spatial dimensions
        attention_scores_flat = attention_scores.view(B, -1)  # (B, H*W)
        attention_weights_flat = F.softmax(attention_scores_flat, dim=1)  # (B, H*W)
        attention_weights = attention_weights_flat.view(B, 1, H, W)  # (B, 1, H, W)
        
        # Apply dropout to attention weights (during training)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of features
        # Reshape feature_map for easier computation
        feature_map_flat = feature_map.view(B, C, H * W)  # (B, C, H*W)
        attention_weights_for_sum = attention_weights_flat.unsqueeze(1)  # (B, 1, H*W)
        
        # Context vector via weighted sum
        context = torch.bmm(feature_map_flat, attention_weights_for_sum.transpose(1, 2))  # (B, C, 1)
        context = context.squeeze(-1)  # (B, C)
        
        if return_attention_weights:
            return context, attention_weights
        else:
            return context, None


class AttentionWorkingMemoryModel(nn.Module):
    """
    Working Memory Model with Task-Guided Attention.
    
    Architecture:
    1. Perceptual Module (CNN): Extracts spatial feature map
    2. Task-Guided Attention: Focuses on task-relevant spatial locations
    3. Cognitive Module (RNN): Processes attended features over time
    4. Classifier: Predicts responses
    
    This model differs from the baseline by inserting an attention mechanism
    between the perceptual and cognitive modules, allowing task-specific
    spatial focus.
    
    Args:
        perceptual: Perceptual module (ResNet50-based)
        cognitive: Cognitive module (RNN/GRU/LSTM)
        hidden_size: Hidden size for RNN
        attention_hidden_dim: Hidden dimension for attention computation
        attention_dropout: Dropout rate for attention
    """
    
    def __init__(
        self,
        perceptual: PerceptualModule,
        cognitive: CognitiveModule,
        hidden_size: int,
        attention_hidden_dim: Optional[int] = None,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.perceptual = perceptual
        self.hidden_size = hidden_size
        
        # Task-guided attention module
        self.attention = TaskGuidedAttention(
            feature_dim=hidden_size,  # After perceptual 1x1 conv reduction
            task_dim=3,
            hidden_dim=attention_hidden_dim or hidden_size,
            dropout=attention_dropout,
        )
        
        self.cognitive = cognitive
        
        # Classifier: hidden state -> 3 response classes
        self.classifier = nn.Linear(hidden_size, 3)
        
        # Storage for attention weights (for visualization)
        self._last_attention_weights = None
    
    def forward(
        self,
        images: torch.Tensor,
        task_vector: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with task-guided attention.
        
        Args:
            images: Input images (B, T, 3, H, W)
            task_vector: Task identity one-hot (B, 3)
            return_attention: If True, return attention weights for visualization
        
        Returns:
            logits: Response logits (B, T, 3)
            hidden_seq: Hidden states per timestep (B, T, H)
            final_state: RNN final state
            attention_weights: Attention weights (B, T, 1, H', W') if return_attention=True
        """
        B, T = images.shape[0], images.shape[1]
        
        # Flatten time into batch for perceptual processing
        x = images.reshape(B * T, *images.shape[2:])  # (B*T, 3, H, W)
        
        # Get spatial feature maps (not pooled yet)
        _, feature_maps = self.perceptual(x, return_feature_map=True)  # (B*T, C, H', W')
        
        if feature_maps is None:
            raise RuntimeError("Perceptual module must return feature maps for attention")
        
        # Reshape to separate batch and time
        BT, C, H_feat, W_feat = feature_maps.shape
        feature_maps = feature_maps.view(B, T, C, H_feat, W_feat)
        
        # Expand task vector across time
        task_rep = task_vector.unsqueeze(1).expand(B, T, 3)  # (B, T, 3)
        
        # Apply task-guided attention at each timestep
        attended_features = []
        attention_weights_list = [] if return_attention else None
        
        for t in range(T):
            feat_t = feature_maps[:, t]  # (B, C, H', W')
            task_t = task_rep[:, t]      # (B, 3)
            
            context_t, attn_t = self.attention(
                feat_t, task_t, return_attention_weights=return_attention
            )
            attended_features.append(context_t)
            
            if return_attention:
                attention_weights_list.append(attn_t)
        
        # Stack attended features
        attended_seq = torch.stack(attended_features, dim=1)  # (B, T, C)
        
        # Concatenate attended features with task vector
        task_rep_for_rnn = task_vector.unsqueeze(1).expand(B, T, 3)
        cog_in = torch.cat([attended_seq, task_rep_for_rnn], dim=-1)  # (B, T, C+3)
        
        # Process through cognitive module
        outputs, final_state, hidden_seq = self.cognitive(cog_in)
        
        # Classify
        logits = self.classifier(outputs)  # (B, T, 3)
        
        # Store attention weights for later visualization
        if return_attention:
            self._last_attention_weights = torch.stack(attention_weights_list, dim=1)  # (B, T, 1, H', W')
            return logits, hidden_seq, final_state, self._last_attention_weights
        else:
            return logits, hidden_seq, final_state, None
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        task_vector: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Prediction with optional attention visualization.
        
        Args:
            images: Input images (B, T, 3, H, W)
            task_vector: Task identity (B, 3)
            return_attention: Return attention weights
        
        Returns:
            preds: Predicted classes (B, T)
            probs: Prediction probabilities (B, T, 3)
            hidden_seq: Hidden states (B, T, H)
            attention_weights: Attention weights if return_attention=True
        """
        logits, hidden_seq, _, attention_weights = self.forward(
            images, task_vector, return_attention=return_attention
        )
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)  # (B, T)
        
        if return_attention:
            return preds, probs, hidden_seq, attention_weights
        else:
            return preds, probs, hidden_seq, None
    
    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass (for visualization)."""
        return self._last_attention_weights
