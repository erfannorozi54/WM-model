"""Adaptation strategies for meta-learning."""

from typing import Dict, Any
import torch.nn as nn


ADAPTATION_METHODS = {
    "scratch": "Train new model from random initialization",
    "full_finetune": "Fine-tune all parameters",
    "attention_only": "Freeze perceptual+cognitive, update attention only",
    "attention_classifier": "Freeze perceptual+cognitive, update attention+classifier",
    "cognitive_only": "Freeze perceptual+attention, update cognitive only",
    "classifier_only": "Freeze all except classifier head",
}


def apply_adaptation_method(model: nn.Module, method: str) -> Dict[str, Any]:
    """Apply parameter freezing strategy based on adaptation method."""
    info = {
        "method": method,
        "trainable_params": 0,
        "frozen_params": 0,
        "trainable_modules": [],
    }
    
    for param in model.parameters():
        param.requires_grad = True
    
    if method in ["scratch", "full_finetune"]:
        pass
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
