"""Adaptation strategies for meta-learning."""

from typing import Dict, Any
import torch.nn as nn


ADAPTATION_METHODS = {
    "scratch": "Train new model from random initialization (CNN frozen)",
    "full_finetune": "Fine-tune all parameters except CNN (perceptual always frozen)",
    "attention_only": "Freeze perceptual+cognitive, update attention only",
    "attention_classifier": "Freeze perceptual+cognitive, update attention+classifier",
    "cognitive_only": "Freeze perceptual+attention, update cognitive only",
    "classifier_only": "Freeze all except classifier head",
}


def apply_adaptation_method(model: nn.Module, method: str) -> Dict[str, Any]:
    """Apply parameter freezing strategy based on adaptation method.
    
    Note: The perceptual (CNN) module is ALWAYS frozen regardless of method.
    """
    info = {
        "method": method,
        "trainable_params": 0,
        "frozen_params": 0,
        "trainable_modules": [],
    }
    
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Always freeze perceptual (CNN) - it's already frozen from model creation
    # but we explicitly ensure it here
    for name, param in model.named_parameters():
        if "perceptual" in name:
            param.requires_grad = False
    
    if method == "scratch":
        # For scratch, only train cognitive and classifier (not perceptual)
        for name, param in model.named_parameters():
            if "perceptual" not in name:
                param.requires_grad = True
    elif method == "full_finetune":
        # Fine-tune all except perceptual (CNN)
        for name, param in model.named_parameters():
            param.requires_grad = "perceptual" not in name
    elif method == "attention_only":
        for name, param in model.named_parameters():
            param.requires_grad = "attention" in name
            if param.requires_grad:
                info["trainable_modules"].append(name)
    elif method == "attention_classifier":
        for name, param in model.named_parameters():
            param.requires_grad = "attention" in name or "classifier" in name
            if param.requires_grad:
                info["trainable_modules"].append(name)
    elif method == "cognitive_only":
        for name, param in model.named_parameters():
            param.requires_grad = "cognitive" in name
            if param.requires_grad:
                info["trainable_modules"].append(name)
    elif method == "classifier_only":
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name
            if param.requires_grad:
                info["trainable_modules"].append(name)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    for param in model.parameters():
        if param.requires_grad:
            info["trainable_params"] += param.numel()
        else:
            info["frozen_params"] += param.numel()
    
    return info
