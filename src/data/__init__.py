"""
Data pipeline for Working Memory experiments.

This module provides:
- ShapeNet dataset download and organization
- 3D to 2D stimulus rendering
- N-back trial sequence generation
- PyTorch Dataset/DataLoader integration
"""

from .shapenet_downloader import (
    ShapeNetDownloader,
    DEFAULT_CATEGORIES,
    DEFAULT_VIEWING_ANGLES,
    scan_generated_stimuli,
    create_sample_stimulus_data,
)
from .renderer import StimulusRenderer
from .nback_generator import (
    NBackGenerator,
    TaskFeature,
    Trial,
    Sequence,
)
from .dataset import (
    NBackDataset,
    NBackDataModule,
)

__all__ = [
    # Always available
    "ShapeNetDownloader",
    "DEFAULT_CATEGORIES",
    "DEFAULT_VIEWING_ANGLES",
    "scan_generated_stimuli",
    "create_sample_stimulus_data",
    # Renderer
    "StimulusRenderer",
    # N-back Generator
    "NBackGenerator",
    "TaskFeature",
    "Trial",
    "Sequence",
    # Dataset
    "NBackDataset",
    "NBackDataModule",
]
