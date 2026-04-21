"""Meta-learning for rapid task adaptation in working memory models."""

from .tasks import NOVEL_TASKS, generate_novel_sequences
from .adaptation import apply_adaptation_method, ADAPTATION_METHODS
from .training import train_epoch, evaluate
from .experiment import run_meta_learning_experiment

__all__ = [
    "NOVEL_TASKS",
    "ADAPTATION_METHODS",
    "generate_novel_sequences",
    "apply_adaptation_method",
    "train_epoch",
    "evaluate",
    "run_meta_learning_experiment",
]
