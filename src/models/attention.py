"""
Task-Guided Feature-Channel Attention Module for Working Memory Model.

This module implements task-guided attention mechanisms that filter
task-irrelevant feature channels based on the task identity vector.

Two attention modes:
1. task_only: Gates computed from task vector only (simpler)
2. dual: Gates computed from both task AND features (more expressive)

The attention learns to gate feature channels based on task relevance,
suppressing task-irrelevant information (distractors) while emphasizing
task-relevant features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from .perceptual import PerceptualModule
from .cognitive import CognitiveModule


class FeatureChannelAttention(nn.Module):
    """
    Task-Guided Feature-Channel Attention Module.
    
    Supports two modes:
    - 'task_only': Gates depend only on task vector (same gates for all inputs)
    - 'dual': Gates depend on both task AND input features (adaptive)
    
    Args:
        feature_dim: Dimension of feature vector (C)
        task_dim: Dimension of task vector (6: 3 feature + 3 N)
        hidden_dim: Hidden dimension for gate computation
        dropout: Dropout rate
        attention_mode: 'task_only' or 'dual'
    """
    
    def __init__(
        self,
        feature_dim: int,
        task_dim: int = 6,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_mode: str = 'task_only',
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim or feature_dim
        self.attention_mode = attention_mode
        
        if attention_mode == 'task_only':
            self._build_task_only_attention(dropout)
        elif attention_mode == 'dual':
            self._build_dual_attention(dropout)
        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")
    
    def _build_task_only_attention(self, dropout: float):
        """Task-only: gates = f(task_vector)"""
        self.gate_network = nn.Sequential(
            nn.Linear(self.task_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.task_bias = nn.Parameter(torch.zeros(self.task_dim, self.feature_dim))
    
    def _build_dual_attention(self, dropout: float):
        """Dual: gates = f(task_vector, features)"""
        # Project task to query space
        self.task_proj = nn.Sequential(
            nn.Linear(self.task_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Project features to key space
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Compute gates from combined representation
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        
    def forward(
        self,
        features: torch.Tensor,
        task_vector: torch.Tensor,
        return_gates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply task-guided channel gating.
        
        Args:
            features: (B, C) or (B, T, C)
            task_vector: (B, 6) - [feature(3), n(3)]
            return_gates: If True, return gate values
        
        Returns:
            gated_features: same shape as input
            gates: gate values if return_gates=True
        """
        if features.dim() == 3:
            B, T, C = features.shape
            task_expanded = task_vector.unsqueeze(1).expand(B, T, -1)
            features_flat = features.reshape(B * T, C)
            task_flat = task_expanded.reshape(B * T, -1)
            
            gated_flat, gates_flat = self._compute_gates(features_flat, task_flat)
            
            gated_features = gated_flat.reshape(B, T, C)
            gates = gates_flat.reshape(B, T, C) if return_gates else None
        else:
            gated_features, gates = self._compute_gates(features, task_vector)
            if not return_gates:
                gates = None
        
        return gated_features, gates
    
    def _compute_gates(
        self,
        features: torch.Tensor,
        task_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core gate computation based on attention mode."""
        
        if self.attention_mode == 'task_only':
            # Gates depend only on task
            gate_logits = self.gate_network(task_vector)
            task_bias = torch.matmul(task_vector, self.task_bias)
            gate_logits = gate_logits + task_bias
            gates = torch.sigmoid(gate_logits)
            
        elif self.attention_mode == 'dual':
            # Gates depend on both task and features
            task_query = self.task_proj(task_vector)      # (B, H)
            feature_key = self.feature_proj(features)     # (B, H)
            combined = task_query * feature_key           # (B, H)
            gate_logits = self.gate_network(combined)     # (B, C)
            gates = torch.sigmoid(gate_logits)
        
        gated_features = features * gates
        return gated_features, gates


class AttentionWorkingMemoryModel(nn.Module):
    """
    Working Memory Model with Task-Guided Feature-Channel Attention.
    
    Args:
        perceptual: Perceptual module (ResNet50-based)
        cognitive: Cognitive module (RNN/GRU/LSTM)
        hidden_size: Hidden size for RNN
        attention_hidden_dim: Hidden dimension for attention
        attention_dropout: Dropout rate for attention
        attention_mode: 'task_only' or 'dual'
    """
    
    def __init__(
        self,
        perceptual: PerceptualModule,
        cognitive: CognitiveModule,
        hidden_size: int,
        attention_hidden_dim: Optional[int] = None,
        attention_dropout: float = 0.1,
        attention_mode: str = 'task_only',
        classifier_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.perceptual = perceptual
        self.hidden_size = hidden_size
        self.attention_mode = attention_mode
        
        # Feature-channel attention module
        self.attention = FeatureChannelAttention(
            feature_dim=hidden_size,
            task_dim=6,  # 3 feature + 3 N
            hidden_dim=attention_hidden_dim or hidden_size,
            dropout=attention_dropout,
            attention_mode=attention_mode,
        )
        
        self.cognitive = cognitive
        
        # Classifier
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
        
        self._last_attention_gates = None
    
    def forward(
        self,
        images: torch.Tensor,
        task_vector: torch.Tensor,
        return_attention: bool = False,
        return_cnn_activations: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T = images.shape[0], images.shape[1]
        
        # Perceptual processing
        x = images.reshape(B * T, *images.shape[2:])
        cnn_features, _ = self.perceptual(x, return_feature_map=False)
        cnn_features = cnn_features.view(B, T, -1)
        
        cnn_activations = cnn_features.clone() if return_cnn_activations else None
        
        # Feature-channel attention
        gated_features, gates = self.attention(
            cnn_features, task_vector, return_gates=return_attention
        )
        
        if return_attention:
            self._last_attention_gates = gates
        
        # Cognitive processing
        task_dim = task_vector.shape[-1]
        task_rep = task_vector.unsqueeze(1).expand(B, T, task_dim)
        cog_in = torch.cat([gated_features, task_rep], dim=-1)
        outputs, final_state, hidden_seq = self.cognitive(cog_in)
        
        # Classification
        logits = self.classifier(outputs)
        
        if return_cnn_activations:
            return logits, hidden_seq, final_state, cnn_activations
        elif return_attention:
            return logits, hidden_seq, final_state, gates
        else:
            return logits, hidden_seq, final_state, None
    
    def get_last_attention_gates(self) -> Optional[torch.Tensor]:
        return self._last_attention_gates
