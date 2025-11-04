"""
Model Factory for Working Memory Models.

This module provides a unified interface for creating different model architectures,
enabling easy comparison between baseline and attention-enhanced models.

Supported architectures:
- Baseline: vanilla RNN, GRU, LSTM
- Attention: Attention-enhanced variants (attention_rnn, attention_gru, attention_lstm)

Usage:
    model = create_model(
        model_type='attention_gru',
        hidden_size=512,
        num_layers=1,
        pretrained_backbone=True,
    )
"""

from typing import Dict, Any, Optional, List
import torch.nn as nn

from .perceptual import PerceptualModule
from .cognitive import VanillaRNN, GRUCog, LSTMCog, CognitiveModule
from .wm_model import WorkingMemoryModel
from .attention import AttentionWorkingMemoryModel


# Mapping of RNN type names to cognitive module classes
COGNITIVE_MODULES = {
    'rnn': VanillaRNN,
    'gru': GRUCog,
    'lstm': LSTMCog,
}


def create_cognitive_module(
    rnn_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    dropout: float = 0.0,
) -> CognitiveModule:
    """
    Create a cognitive module (RNN variant).
    
    Args:
        rnn_type: Type of RNN ('rnn', 'gru', 'lstm')
        input_size: Input dimension
        hidden_size: Hidden state dimension
        num_layers: Number of RNN layers
        dropout: Dropout rate
    
    Returns:
        Cognitive module instance
    """
    rnn_type = rnn_type.lower()
    
    if rnn_type not in COGNITIVE_MODULES:
        raise ValueError(f"Unknown RNN type: {rnn_type}. Choose from {list(COGNITIVE_MODULES.keys())}")
    
    cognitive_class = COGNITIVE_MODULES[rnn_type]
    
    return cognitive_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def create_model(
    model_type: str = 'gru',
    hidden_size: int = 512,
    num_layers: int = 1,
    dropout: float = 0.0,
    pretrained_backbone: bool = True,
    freeze_backbone: bool = True,
    capture_exact_layer42_relu: bool = True,
    attention_hidden_dim: Optional[int] = None,
    attention_dropout: float = 0.1,
    classifier_layers: Optional[List[int]] = None,
) -> nn.Module:
    """
    Factory function to create working memory models.
    
    Args:
        model_type: Type of model. Options:
            - 'rnn', 'gru', 'lstm': Baseline models
            - 'attention_rnn', 'attention_gru', 'attention_lstm': Attention-enhanced models
        hidden_size: Hidden state dimension for RNN
        num_layers: Number of RNN layers
        dropout: RNN dropout rate
        pretrained_backbone: Use pretrained ResNet50
        freeze_backbone: Freeze backbone weights
        capture_exact_layer42_relu: Capture exact layer4[2].relu activation
        attention_hidden_dim: Hidden dimension for attention (attention models only)
        attention_dropout: Dropout rate for attention (attention models only)
    
    Returns:
        Model instance (WorkingMemoryModel or AttentionWorkingMemoryModel)
    """
    model_type = model_type.lower()
    
    # Determine if this is an attention model
    is_attention = model_type.startswith('attention_')
    
    # Extract base RNN type
    if is_attention:
        rnn_type = model_type.replace('attention_', '')
    else:
        rnn_type = model_type
    
    # Validate RNN type
    if rnn_type not in COGNITIVE_MODULES:
        raise ValueError(
            f"Unknown RNN type: {rnn_type}. "
            f"Valid options: {list(COGNITIVE_MODULES.keys())} or "
            f"attention_{{{', '.join(COGNITIVE_MODULES.keys())}}}"
        )
    
    # Create perceptual module
    perceptual = PerceptualModule(
        out_channels=hidden_size,
        pretrained=pretrained_backbone,
        freeze_backbone=freeze_backbone,
        capture_exact_layer42_relu=capture_exact_layer42_relu,
    )
    
    # Create cognitive module
    # Input size = hidden_size + 3 (task vector)
    cognitive = create_cognitive_module(
        rnn_type=rnn_type,
        input_size=hidden_size + 3,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    # Create full model based on type
    if is_attention:
        # Attention-enhanced model
        model = AttentionWorkingMemoryModel(
            perceptual=perceptual,
            cognitive=cognitive,
            hidden_size=hidden_size,
            attention_hidden_dim=attention_hidden_dim,
            attention_dropout=attention_dropout,
            classifier_layers=classifier_layers,
        )
    else:
        # Baseline model
        model = WorkingMemoryModel(
            perceptual=perceptual,
            cognitive=cognitive,
            hidden_size=hidden_size,
            classifier_layers=classifier_layers,
        )
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: Model instance
    
    Returns:
        Dictionary with model information
    """
    from .wm_model import WorkingMemoryModel
    from .attention import AttentionWorkingMemoryModel
    
    info = {
        'type': type(model).__name__,
        'is_attention': isinstance(model, AttentionWorkingMemoryModel),
        'is_baseline': isinstance(model, WorkingMemoryModel) and not isinstance(model, AttentionWorkingMemoryModel),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # Get RNN type from cognitive module
    if hasattr(model, 'cognitive'):
        cog_type = type(model.cognitive).__name__
        if 'RNN' in cog_type or 'Vanilla' in cog_type:
            info['rnn_type'] = 'rnn'
        elif 'GRU' in cog_type:
            info['rnn_type'] = 'gru'
        elif 'LSTM' in cog_type:
            info['rnn_type'] = 'lstm'
        else:
            info['rnn_type'] = 'unknown'
    
    return info


def print_model_summary(model: nn.Module):
    """
    Print a summary of the model architecture.
    
    Args:
        model: Model instance
    """
    info = get_model_info(model)
    
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Model Type: {info['type']}")
    print(f"Architecture: {'Attention-Enhanced' if info['is_attention'] else 'Baseline'}")
    if 'rnn_type' in info:
        print(f"RNN Type: {info['rnn_type'].upper()}")
    print(f"Total Parameters: {info['num_parameters']:,}")
    print(f"Trainable Parameters: {info['num_trainable_parameters']:,}")
    
    # Print module breakdown
    print("\nModule Breakdown:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params:,} parameters")
    
    print("=" * 70)


# Convenience functions for specific model types
def create_baseline_model(rnn_type: str = 'gru', **kwargs) -> WorkingMemoryModel:
    """Create a baseline working memory model."""
    return create_model(model_type=rnn_type, **kwargs)


def create_attention_model(rnn_type: str = 'gru', **kwargs) -> AttentionWorkingMemoryModel:
    """Create an attention-enhanced working memory model."""
    return create_model(model_type=f'attention_{rnn_type}', **kwargs)


if __name__ == "__main__":
    # Demo: Create and compare different models
    print("\n" + "=" * 70)
    print("MODEL FACTORY DEMO")
    print("=" * 70)
    
    # Create baseline GRU
    print("\n1. Baseline GRU Model:")
    baseline_gru = create_baseline_model('gru', hidden_size=512)
    print_model_summary(baseline_gru)
    
    # Create attention GRU
    print("\n2. Attention-Enhanced GRU Model:")
    attention_gru = create_attention_model('gru', hidden_size=512)
    print_model_summary(attention_gru)
    
    # Compare parameter counts
    print("\n3. Parameter Comparison:")
    baseline_info = get_model_info(baseline_gru)
    attention_info = get_model_info(attention_gru)
    
    param_diff = attention_info['num_parameters'] - baseline_info['num_parameters']
    param_ratio = attention_info['num_parameters'] / baseline_info['num_parameters']
    
    print(f"Baseline parameters: {baseline_info['num_parameters']:,}")
    print(f"Attention parameters: {attention_info['num_parameters']:,}")
    print(f"Additional parameters: {param_diff:,} (+{(param_ratio-1)*100:.1f}%)")
    
    print("\n" + "=" * 70)
