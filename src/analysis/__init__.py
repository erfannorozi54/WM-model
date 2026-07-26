# Analysis package for Phase 3 (activations, decoding, orthogonalization)

import os

# Bootstrap-heavy modules (neural_efficiency.py, gate_suppression.py, compare_models.py)
# call numpy/sklearn linear algebra (eigvalsh, LinearSVC, ...) thousands of times on small
# matrices inside 1000-iteration resampling loops. Left at its default, OpenBLAS/MKL spawns
# a full thread pool (up to os.cpu_count()) for EACH call; at thousands of calls per run,
# and especially with multiple such analyses or GPU training jobs sharing the same machine,
# this causes catastrophic thread-contention (observed: load average ~100 on a 64-core box,
# a ~1-2 minute analysis taking 4+ hours) rather than any real compute cost. These env vars
# must be set before numpy/torch/sklearn are imported anywhere (BLAS reads them once at
# first use), so this must stay the first thing this package does.
for _threads_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads_var, "1")

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
from .neural_efficiency import (
    activation_magnitude,
    participation_ratio,
    population_sparsity,
    fano_factor_analogue,
    compare_cell,
    run_neural_efficiency_analysis,
    select_matched_epoch,
)
from .gate_suppression import (
    channel_relevance_scores,
    gate_suppression_index,
    compare_gate_suppression,
)
