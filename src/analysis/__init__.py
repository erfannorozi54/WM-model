# Analysis package for Phase 3 (activations, decoding, orthogonalization)

from .activations import load_payloads, build_matrix, build_cnn_matrix
from .decoding import train_decoder
from .orthogonalization import one_vs_rest_weights, orthogonalization_index
from .procrustes import (
    compute_procrustes_alignment,
    reconstruct_weights,
    evaluate_reconstruction,
    procrustes_analysis,
    swap_hypothesis_test,
)
