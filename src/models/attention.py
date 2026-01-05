"""
Task-Guided Feature-Channel Attention Module for Working Memory Model.

This module implements a task-guided attention mechanism that filters
task-irrelevant feature channels based on the task identity vector.

Key Insight:
- Location task: needs spatial/positional features
- Identity task: needs object-specific features  
- Category task: needs semantic/categorical features

The attention learns to gate feature channels based on task relevance,
suppressing task-irrelevant information (distractors) while emphasizing
task-relevant features.

Architecture:
1. CNN extracts features (B, C) containing all object properties
2. Task vector (B, 3) specifies which property is relevant
3. Feature-Channel Attention computes per-channel gates
4. Gated features are passed to RNN for temporal processing
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
    
    This module computes channel-wise attention gates based on the task vector,
    allowing the model to filter out task-irrelevant features and emphasize
    task-relevant ones.
    
    The key idea: different feature channels encode different object properties
    (location, identity, category). The task vector tells us which property
    matters, so we learn to gate channels accordingly.
    
    Architecture:
        task_vector (B, 3) → MLP → channel_gates (B, C) ∈ [0, 1]
        output = features * channel_gates  (element-wise)
    
    Args:
        feature_dim: Dimension of feature vector (C)
        task_dim: Dimension of task vector (3 for location/identity/category)
        hidden_dim: Hidden dimension for gate computation
        dropout: Dropout rate
        gate_activation: 'sigmoid' for soft gates, 'softmax' for competitive
    """
    
    def __init__(
        self,
        feature_dim: int,
        task_dim: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        gate_activation: str = 'sigmoid',
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim or feature_dim
        self.gate_activation = gate_activation
        
        # Gate network: task → channel gates
        # Uses task vector to compute which channels are relevant
        self.gate_network = nn.Sequential(
            nn.Linear(task_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, feature_dim),
        )
        
        # Learnable bias for each task (optional enhancement)
        # This allows task-specific baseline activations
        self.task_bias = nn.Parameter(torch.zeros(task_dim, feature_dim))
        
        # Temperature for softmax (if used)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self,
        features: torch.Tensor,
        task_vector: torch.Tensor,
        return_gates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply task-guided channel gating to features.
        
        Args:
            features: Feature vector (B, C) or (B, T, C)
            task_vector: Task identity one-hot (B, 3)
            return_gates: If True, return gate values for analysis
        
        Returns:
            gated_features: Task-filtered features, same shape as input
            gates: Channel gate values (B, C) if return_gates=True
        """
        # Handle both (B, C) and (B, T, C) inputs
        if features.dim() == 3:
            B, T, C = features.shape
            # Expand task vector for all timesteps
            task_expanded = task_vector.unsqueeze(1).expand(B, T, -1)  # (B, T, 3)
            # Reshape for batch processing
            features_flat = features.reshape(B * T, C)
            task_flat = task_expanded.reshape(B * T, -1)
            
            gated_flat, gates_flat = self._apply_gates(features_flat, task_flat)
            
            gated_features = gated_flat.reshape(B, T, C)
            gates = gates_flat.reshape(B, T, C) if return_gates else None
        else:
            gated_features, gates = self._apply_gates(features, task_vector)
        
        if return_gates:
            return gated_features, gates
        return gated_features, None
    
    def _apply_gates(
        self,
        features: torch.Tensor,
        task_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core gating computation.
        
        Args:
            features: (B, C)
            task_vector: (B, 3)
        
        Returns:
            gated_features: (B, C)
            gates: (B, C)
        """
        # Compute base gates from task
        gate_logits = self.gate_network(task_vector)  # (B, C)
        
        # Add task-specific bias
        # task_vector is one-hot, so this selects the bias for the active task
        task_bias = torch.matmul(task_vector, self.task_bias)  # (B, C)
        gate_logits = gate_logits + task_bias
        
        # Apply activation to get gates in [0, 1]
        if self.gate_activation == 'sigmoid':
            gates = torch.sigmoid(gate_logits)
        elif self.gate_activation == 'softmax':
            # Softmax makes channels compete (sum to 1)
            gates = F.softmax(gate_logits / self.temperature, dim=-1) * self.feature_dim
        else:
            gates = torch.sigmoid(gate_logits)
        
        # Apply gates to features
        gated_features = features * gates
        
        return gated_features, gates


class AttentionWorkingMemoryModel(nn.Module):
    """
    Working Memory Model with Task-Guided Feature-Channel Attention.
    
    This model filters task-irrelevant features using channel attention,
    allowing the RNN to focus on task-relevant information.
    
    Architecture:
    1. Perceptual Module (CNN): Extracts visual features (B, T, C)
    2. Feature-Channel Attention: Gates channels by task relevance
    3. Cognitive Module (RNN): Processes filtered features over time
    4. Classifier: Predicts responses
    
    Key difference from baseline:
    - Baseline: All features (location + identity + category) go to RNN
    - Attention: Task-irrelevant features are suppressed before RNN
    
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
        classifier_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.perceptual = perceptual
        self.hidden_size = hidden_size
        
        # Feature-channel attention module
        self.attention = FeatureChannelAttention(
            feature_dim=hidden_size,
            task_dim=3,
            hidden_dim=attention_hidden_dim or hidden_size,
            dropout=attention_dropout,
            gate_activation='sigmoid',
        )
        
        self.cognitive = cognitive
        
        # Classifier: hidden state -> 3 response classes
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
        
        # Storage for attention gates (for analysis)
        self._last_attention_gates = None
    
    def forward(
        self,
        images: torch.Tensor,
        task_vector: torch.Tensor,
        return_attention: bool = False,
        return_cnn_activations: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with task-guided feature-channel attention.
        
        Args:
            images: Input images (B, T, 3, H, W)
            task_vector: Task identity one-hot (B, 3)
            return_attention: If True, return attention gate values
            return_cnn_activations: If True, return CNN activations (for analysis)
        
        Returns:
            logits: Response logits (B, T, 3)
            hidden_seq: Hidden states per timestep (B, T, H)
            final_state: RNN final state
            attention_gates: Channel gate values (B, T, C) if return_attention=True
        """
        B, T = images.shape[0], images.shape[1]
        
        # Flatten time into batch for perceptual processing
        x = images.reshape(B * T, *images.shape[2:])  # (B*T, 3, H, W)
        
        # Get CNN features (pooled)
        cnn_features, _ = self.perceptual(x, return_feature_map=False)  # (B*T, C)
        
        # Reshape to separate batch and time
        cnn_features = cnn_features.view(B, T, -1)  # (B, T, C)
        
        # Store CNN activations before attention (for analysis)
        cnn_activations = cnn_features.clone() if return_cnn_activations else None
        
        # Apply task-guided feature-channel attention
        gated_features, gates = self.attention(
            cnn_features, task_vector, return_gates=return_attention
        )  # (B, T, C)
        
        # Store gates for later analysis
        if return_attention:
            self._last_attention_gates = gates
        
        # Concatenate gated features with task vector
        task_rep = task_vector.unsqueeze(1).expand(B, T, 3)  # (B, T, 3)
        cog_in = torch.cat([gated_features, task_rep], dim=-1)  # (B, T, C+3)
        
        # Process through cognitive module
        outputs, final_state, hidden_seq = self.cognitive(cog_in)
        
        # Classify
        logits = self.classifier(outputs)  # (B, T, 3)
        
        if return_cnn_activations:
            return logits, hidden_seq, final_state, cnn_activations
        elif return_attention:
            return logits, hidden_seq, final_state, gates
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
            return_attention: Return attention gate values
        
        Returns:
            preds: Predicted classes (B, T)
            probs: Prediction probabilities (B, T, 3)
            hidden_seq: Hidden states (B, T, H)
            attention_gates: Gate values if return_attention=True
        """
        logits, hidden_seq, _, attention_gates = self.forward(
            images, task_vector, return_attention=return_attention
        )
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)  # (B, T)
        
        if return_attention:
            return preds, probs, hidden_seq, attention_gates
        else:
            return preds, probs, hidden_seq, None
    
    def get_last_attention_gates(self) -> Optional[torch.Tensor]:
        """Get attention gates from last forward pass (for analysis)."""
        return self._last_attention_gates
    
    def analyze_gate_statistics(self, task_vector: torch.Tensor) -> dict:
        """
        Analyze what the attention gates look like for each task.
        Useful for understanding if the model learned task-specific filtering.
        
        Args:
            task_vector: One-hot task vector (B, 3)
        
        Returns:
            Dictionary with gate statistics
        """
        # Create dummy features to see gate patterns
        dummy_features = torch.ones(1, self.hidden_size)
        
        stats = {}
        task_names = ['location', 'identity', 'category']
        
        for i, name in enumerate(task_names):
            task = torch.zeros(1, 3)
            task[0, i] = 1.0
            
            _, gates = self.attention(dummy_features, task, return_gates=True)
            gates = gates.squeeze()
            
            stats[name] = {
                'mean_gate': gates.mean().item(),
                'std_gate': gates.std().item(),
                'min_gate': gates.min().item(),
                'max_gate': gates.max().item(),
                'active_channels': (gates > 0.5).sum().item(),
                'suppressed_channels': (gates < 0.5).sum().item(),
            }
        
        return stats
