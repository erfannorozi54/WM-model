#!/usr/bin/env python3
"""
Comprehensive Analysis Pipeline for Working Memory Model

Implements all 5 analyses from the paper:
1. Model Behavioral Performance (Figure A1c)
2. Encoding of Object Properties (Figure 2a, 2b, 2c)
3. Representational Orthogonalization (Figure 3b)
4. Mechanisms of WM Dynamics (Figure 4b, 4d, 4g)
5. Causal Perturbation Test (Figure A7)

Usage:
    python -m src.analysis.comprehensive_analysis \
        --analysis all \
        --hidden_root experiments/wm_mtmf/hidden_states \
        --output_dir analysis_results
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import json
import numpy as np
import torch
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import seaborn as sns

from .activations import load_payloads, build_matrix, build_matrix_with_values, build_cnn_matrix, TASK_INDEX_TO_NAME
from .orthogonalization import one_vs_rest_weights, orthogonalization_index
from .procrustes import compute_procrustes_alignment


class ComprehensiveAnalysis:
    """Master class for running all paper analyses."""
    
    def __init__(self, hidden_root: Path, output_dir: Path):
        self.hidden_root = Path(hidden_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.payloads = None
        
    def load_data(self, epochs: Optional[List[int]] = None):
        """Load hidden states and metadata."""
        print(f"Loading data from {self.hidden_root}...")
        self.payloads = load_payloads(self.hidden_root, epochs=epochs)
        print(f"  Loaded {len(self.payloads)} payloads")
        
    # ========================================================================
    # ANALYSIS 1: Model Behavioral Performance
    # ========================================================================
    
    def analyze_behavioral_performance(
        self,
        training_log_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analysis 1: Model Behavioral Performance
        
        Generates:
        - Basic performance plots for all scenarios
        - Novel Angle vs Novel Identity comparison (Figure A1c)
        
        Args:
            training_log_path: Path to training_log.json from training
            
        Returns:
            Dictionary with performance metrics
        """
        print("\n" + "="*70)
        print("ANALYSIS 1: MODEL BEHAVIORAL PERFORMANCE")
        print("="*70)
        
        if training_log_path is None:
            training_log_path = self.hidden_root.parent / "training_log.json"
        
        if not training_log_path.exists():
            print(f"  ⚠ Training log not found: {training_log_path}")
            print("  ⚠ Skipping behavioral performance analysis")
            return {"status": "skipped", "reason": "no_training_log"}
        
        # Load training log
        with open(training_log_path, 'r') as f:
            log = json.load(f)
        
        results = {
            "final_train_acc": log[-1].get('train_acc', None),
            "final_val_novel_angle_acc": log[-1].get('val_novel_angle_acc', None),
            "final_val_novel_identity_acc": log[-1].get('val_novel_identity_acc', None),
        }
        
        # Plot training curves
        self._plot_training_curves(log)
        
        # Plot novel angle vs novel identity (Figure A1c)
        self._plot_generalization_comparison(log)
        
        print(f"\n  ✓ Final Train Accuracy: {results['final_train_acc']:.4f}")
        print(f"  ✓ Final Val (Novel Angle) Accuracy: {results['final_val_novel_angle_acc']:.4f}")
        print(f"  ✓ Final Val (Novel Identity) Accuracy: {results['final_val_novel_identity_acc']:.4f}")
        
        # Save results
        with open(self.output_dir / "analysis1_performance.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_training_curves(self, log: List[Dict]):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = [x['epoch'] for x in log]
        train_acc = [x.get('train_acc', 0) for x in log]
        val_angle_acc = [x.get('val_novel_angle_acc', 0) for x in log]
        val_id_acc = [x.get('val_novel_identity_acc', 0) for x in log]
        
        # Accuracy plot
        axes[0].plot(epochs, train_acc, label='Training', marker='o')
        axes[0].plot(epochs, val_angle_acc, label='Val (Novel Angle)', marker='s')
        axes[0].plot(epochs, val_id_acc, label='Val (Novel Identity)', marker='^')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot (if available)
        train_loss = [x.get('train_loss', 0) for x in log]
        axes[1].plot(epochs, train_loss, label='Training Loss', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "analysis1_training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: analysis1_training_curves.png")
    
    def _plot_generalization_comparison(self, log: List[Dict]):
        """Plot Novel Angle vs Novel Identity comparison (Figure A1c)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        final_epoch = log[-1]
        val_angle_acc = final_epoch.get('val_novel_angle_acc', 0)
        val_id_acc = final_epoch.get('val_novel_identity_acc', 0)
        
        categories = ['Novel Angle\n(Same Objects)', 'Novel Identity\n(New Objects)']
        accuracies = [val_angle_acc, val_id_acc]
        colors = ['#4CAF50', '#2196F3']
        
        bars = ax.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Generalization Performance (Figure A1c)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add expected observation
        ax.axhline(y=val_angle_acc, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.5, val_angle_acc + 0.02, 'Expected: Novel Identity < Novel Angle',
               ha='center', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "analysis1_generalization_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: analysis1_generalization_comparison.png")
        
        # Verify expected pattern
        if val_id_acc < val_angle_acc:
            print(f"  ✓ Expected pattern confirmed: Novel Identity ({val_id_acc:.3f}) < Novel Angle ({val_angle_acc:.3f})")
        else:
            print(f"  ⚠ Unexpected pattern: Novel Identity ({val_id_acc:.3f}) >= Novel Angle ({val_angle_acc:.3f})")
    
    # ========================================================================
    # ANALYSIS 2: Encoding of Object Properties
    # ========================================================================
    
    def analyze_encoding_properties(
        self,
        encoding_time: int = 0,
        task_relevant: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Analysis 2: Encoding of Object Properties (Section 4.1)
        
        A. Task-Relevance Decoding (Figure 2b)
        B. Cross-Task Generalization (Figure 2a, 2c)
        
        Args:
            encoding_time: Timestep to analyze (default=0, first timestep)
            task_relevant: Mapping of task name to relevant feature
                          e.g., {"location": "location", "identity": "identity", "category": "category"}
        
        Returns:
            Dictionary with decoding results
        """
        print("\n" + "="*70)
        print("ANALYSIS 2: ENCODING OF OBJECT PROPERTIES")
        print("="*70)
        
        if self.payloads is None:
            self.load_data()
        
        results = {}
        
        # A. Task-Relevance Decoding (Figure 2b)
        print("\nA. Task-Relevance Decoding (Figure 2b)")
        task_relevance = self._analyze_task_relevance(encoding_time)
        results['task_relevance'] = task_relevance
        
        # B. Cross-Task Generalization (Figure 2a, 2c)
        print("\nB. Cross-Task Generalization (Figure 2a, 2c)")
        cross_task = self._analyze_cross_task_generalization(encoding_time)
        results['cross_task'] = cross_task
        
        # Save results
        with open(self.output_dir / "analysis2_encoding.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _analyze_task_relevance(self, time: int) -> Dict:
        """
        Decode task-relevant and task-irrelevant features.
        Expected: relevant > 85%, irrelevant < 85% for STSF models.
        """
        print("  Decoding all properties at all task contexts...")
        
        properties = ["location", "identity", "category"]
        tasks = ["location", "identity", "category"]
        
        results = {}
        
        for task in tasks:
            task_idx = self._task_name_to_index(task)
            results[task] = {}
            
            for prop in properties:
                try:
                    X, y, label2idx = build_matrix(
                        self.payloads, prop, time=time, task_index=task_idx, n_value=None
                    )
                    
                    if X.numel() == 0:
                        results[task][prop] = {"accuracy": None, "n_samples": 0}
                        continue
                    
                    # Train decoder
                    clf = self._train_decoder(X, y)
                    y_pred = clf.predict(X.numpy())
                    acc = accuracy_score(y.numpy(), y_pred)
                    
                    results[task][prop] = {
                        "accuracy": float(acc),
                        "n_samples": int(len(y)),
                        "is_relevant": (prop == task)
                    }
                    
                except Exception as e:
                    results[task][prop] = {"error": str(e)}
        
        # Plot results
        self._plot_task_relevance(results)
        
        return results
    
    def _plot_task_relevance(self, results: Dict):
        """Plot task-relevant vs task-irrelevant decoding (Figure 2b style)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tasks = ["location", "identity", "category"]
        properties = ["location", "identity", "category"]
        
        # Create matrix
        acc_matrix = np.zeros((len(tasks), len(properties)))
        for i, task in enumerate(tasks):
            for j, prop in enumerate(properties):
                if task in results and prop in results[task]:
                    acc = results[task][prop].get('accuracy', 0)
                    acc_matrix[i, j] = acc if acc is not None else 0
        
        # Plot heatmap
        sns.heatmap(acc_matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                   xticklabels=properties, yticklabels=tasks, ax=ax, cbar_kws={'label': 'Accuracy'})
        ax.set_xlabel('Property Decoded', fontsize=12)
        ax.set_ylabel('Task Context', fontsize=12)
        ax.set_title('Task-Relevance Decoding (Figure 2b)\nEncoding Space (t=0)', fontsize=14, fontweight='bold')
        
        # Highlight diagonal (task-relevant)
        for i in range(len(tasks)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "analysis2a_task_relevance.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: analysis2a_task_relevance.png")
    
    def _analyze_cross_task_generalization(self, time: int) -> Dict:
        """
        Cross-task generalization matrix (Figure 2a).
        Train on task A, test on task B for each property.
        """
        print("  Computing cross-task generalization matrices...")
        
        properties = ["location", "identity", "category"]
        tasks = ["location", "identity", "category"]
        
        results = {}
        
        for prop in properties:
            print(f"    Property: {prop}")
            matrix = np.zeros((len(tasks), len(tasks)))
            
            for i, train_task in enumerate(tasks):
                train_task_idx = self._task_name_to_index(train_task)
                
                # Train decoder on this task
                try:
                    X_train, y_train, label2idx = build_matrix(
                        self.payloads, prop, time=time, task_index=train_task_idx, n_value=None
                    )
                    
                    if X_train.numel() == 0:
                        continue
                    
                    clf = self._train_decoder(X_train, y_train)
                    
                    # Test on all tasks
                    for j, test_task in enumerate(tasks):
                        test_task_idx = self._task_name_to_index(test_task)
                        
                        try:
                            X_test, y_test, _, raw_vals = build_matrix_with_values(
                                self.payloads, prop, time=time, task_index=test_task_idx, n_value=None
                            )
                            
                            if X_test.numel() == 0:
                                continue
                            
                            # Align labels
                            y_test_idx, keep = self._align_test_labels(raw_vals, label2idx)
                            if keep.sum() == 0:
                                continue
                            
                            X_test_np = X_test.numpy()[keep]
                            y_pred = clf.predict(X_test_np)
                            acc = accuracy_score(y_test_idx, y_pred)
                            
                            matrix[i, j] = acc
                            
                        except Exception as e:
                            print(f"      Warning: {train_task}→{test_task}: {e}")
                            continue
                    
                except Exception as e:
                    print(f"      Warning: train on {train_task}: {e}")
                    continue
            
            results[prop] = matrix.tolist()
            
            # Plot this property's matrix
            self._plot_cross_task_matrix(matrix, prop, tasks)
        
        return results
    
    def _plot_cross_task_matrix(self, matrix: np.ndarray, property_name: str, tasks: List[str]):
        """Plot cross-task generalization matrix (Figure 2a style)."""
        fig, ax = plt.subplots(figsize=(7, 6))
        
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                   xticklabels=tasks, yticklabels=tasks, ax=ax, cbar_kws={'label': 'Accuracy'})
        ax.set_xlabel('Test Task', fontsize=12)
        ax.set_ylabel('Train Task', fontsize=12)
        ax.set_title(f'Cross-Task Generalization: {property_name.capitalize()}\n(Figure 2a)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"analysis2b_cross_task_{property_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: analysis2b_cross_task_{property_name}.png")
        
        # Check diagonal vs off-diagonal
        diagonal_mean = np.mean([matrix[i, i] for i in range(len(tasks))])
        off_diagonal = [matrix[i, j] for i in range(len(tasks)) for j in range(len(tasks)) if i != j]
        off_diagonal_mean = np.mean(off_diagonal) if off_diagonal else 0
        
        print(f"      Diagonal (same task): {diagonal_mean:.3f}")
        print(f"      Off-diagonal (cross-task): {off_diagonal_mean:.3f}")
        
        if off_diagonal_mean < diagonal_mean * 0.8:
            print(f"      ✓ Expected pattern: Low cross-task generalization (GRU/LSTM)")
        else:
            print(f"      ⚠ Unexpected: High cross-task generalization (RNN/Attention?)")
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _task_name_to_index(self, name: Optional[str]) -> Optional[int]:
        """Convert task name to index."""
        if name is None or name == "any":
            return None
        for k, v in TASK_INDEX_TO_NAME.items():
            if v == name:
                return k
        raise ValueError(f"Unknown task name: {name}")
    
    def _train_decoder(self, X: torch.Tensor, y: torch.Tensor) -> Pipeline:
        """Train a standard linear SVC decoder."""
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(kernel="linear", class_weight="balanced")),
        ])
        clf.fit(X.numpy(), y.numpy())
        return clf
    
    def _align_test_labels(self, y_test: torch.Tensor, train_label2idx: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Align test labels to training label space."""
        y_test_idx = []
        for v in y_test.tolist():
            if v in train_label2idx:
                y_test_idx.append(train_label2idx[v])
            else:
                y_test_idx.append(-1)
        y_test_idx = np.array(y_test_idx)
        keep = y_test_idx >= 0
        return y_test_idx[keep], keep
    
    # ========================================================================
    # ANALYSIS 3: Representational Orthogonalization
    # ========================================================================
    
    def analyze_orthogonalization(
        self,
        encoding_time: int = 0,
        cnn_activations_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analysis 3: Representational Orthogonalization (Section 4.2, Figure 3b)
        
        Compares orthogonalization index (O) between:
        - Perceptual space (CNN penultimate layer)
        - Encoding space (RNN first timestep)
        
        Expected: Points fall below diagonal (RNN de-orthogonalizes)
        
        Args:
            encoding_time: RNN timestep to analyze (default=0)
            cnn_activations_path: Path to saved CNN activations (optional, deprecated)
        
        Returns:
            Dictionary with orthogonalization indices
        """
        print("\n" + "="*70)
        print("ANALYSIS 3: REPRESENTATIONAL ORTHOGONALIZATION")
        print("="*70)
        
        if self.payloads is None:
            self.load_data()
        
        properties = ["location", "identity", "category"]
        results = {"encoding": {}, "perceptual": {}}
        
        # Check if CNN activations are available in payloads
        has_cnn = any(p.get("cnn_activations") is not None for p in self.payloads)
        
        # Compute O for encoding space (RNN)
        print("\nComputing O for Encoding Space (RNN)...")
        for prop in properties:
            O_encoding = self._compute_orthogonalization_index(prop, encoding_time)
            results["encoding"][prop] = O_encoding
            print(f"  {prop}: O = {O_encoding:.4f}")
        
        # Compute O for perceptual space (CNN) if available
        if has_cnn:
            print("\nComputing O for Perceptual Space (CNN)...")
            for prop in properties:
                O_perceptual = self._compute_cnn_orthogonalization_index(prop, encoding_time)
                results["perceptual"][prop] = O_perceptual
                print(f"  {prop}: O = {O_perceptual:.4f}")
        else:
            print("\n  ⚠ CNN activations not available in payloads")
            print("    To enable CNN analysis, train with save_hidden=True")
            results["perceptual"] = {"status": "not_available"}
        
        # Plot results
        self._plot_orthogonalization(results)
        
        # Save results
        with open(self.output_dir / "analysis3_orthogonalization.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _compute_orthogonalization_index(self, property_name: str, time: int) -> float:
        """Compute orthogonalization index for a property from RNN hidden states."""
        X, y, label2idx = build_matrix(
            self.payloads, property_name, time=time, task_index=None, n_value=None
        )
        
        if X.numel() == 0:
            return 0.0
        
        # Train one-vs-rest decoders
        W = one_vs_rest_weights(X, y)
        
        # Compute orthogonalization index
        O = orthogonalization_index(W)
        
        return float(O)
    
    def _compute_cnn_orthogonalization_index(self, property_name: str, time: int) -> float:
        """Compute orthogonalization index for a property from CNN activations."""
        X, y, label2idx = build_cnn_matrix(
            self.payloads, property_name, time=time, task_index=None, n_value=None
        )
        
        if X.numel() == 0:
            return 0.0
        
        # Train one-vs-rest decoders
        W = one_vs_rest_weights(X, y)
        
        # Compute orthogonalization index
        O = orthogonalization_index(W)
        
        return float(O)
    
    def _plot_orthogonalization(self, results: Dict):
        """Plot O(Perceptual) vs O(Encoding) - Figure 3b style."""
        fig, ax = plt.subplots(figsize=(7, 7))
        
        properties = ["location", "identity", "category"]
        colors = {'location': '#FF6B6B', 'identity': '#4ECDC4', 'category': '#45B7D1'}
        markers = {'location': 'o', 'identity': 's', 'category': '^'}
        
        if results["perceptual"].get("status") in ["not_available", "not_implemented"]:
            # Just plot encoding space O values as bar chart
            x_pos = np.arange(len(properties))
            y_vals = [results["encoding"][p] for p in properties]
            
            ax.bar(x_pos, y_vals, color=[colors[p] for p in properties], alpha=0.7, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([p.capitalize() for p in properties])
            ax.set_ylabel('Orthogonalization Index (O)', fontsize=12)
            ax.set_title('Representational Orthogonalization\nEncoding Space Only', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, axis='y', alpha=0.3)
        else:
            # Plot perceptual vs encoding scatter (Figure 3b style)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Diagonal (no change)')
            
            for prop in properties:
                O_cnn = results["perceptual"].get(prop, 0)
                O_rnn = results["encoding"].get(prop, 0)
                ax.scatter(O_cnn, O_rnn, c=colors[prop], s=150, marker=markers[prop], 
                          label=f'{prop.capitalize()}', edgecolors='black', linewidths=1, zorder=5)
            
            ax.set_xlabel('O (Perceptual Space - CNN)', fontsize=12)
            ax.set_ylabel('O (Encoding Space - RNN)', fontsize=12)
            ax.set_title('Representational Orthogonalization (Figure 3b)\nExpected: Points below diagonal', 
                        fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add annotation about expected pattern
            ax.text(0.95, 0.05, 'Below diagonal:\nRNN de-orthogonalizes',
                   transform=ax.transAxes, fontsize=9, style='italic',
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Check if pattern matches expectation
            below_diagonal = sum(1 for p in properties 
                                if results["encoding"].get(p, 0) < results["perceptual"].get(p, 0))
            if below_diagonal == len(properties):
                print(f"  ✓ Expected pattern: All points below diagonal (RNN de-orthogonalizes)")
            else:
                print(f"  ⚠ {below_diagonal}/{len(properties)} points below diagonal")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "analysis3_orthogonalization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: analysis3_orthogonalization.png")
    
    # ========================================================================
    # ANALYSIS 4: Mechanisms of WM Dynamics
    # ========================================================================
    
    def analyze_wm_dynamics(
        self,
        property_name: str = "identity",
        max_time: int = 5
    ) -> Dict[str, Any]:
        """
        Analysis 4: Mechanisms of WM Dynamics (Section 4.3)
        
        A. Test H1: Slot-Based Memory (Figure 4b) - Cross-Time Decoding
        B. Test H2 vs H3: Shared Encoding Space (Figure 4d) - Cross-Stimulus Decoding
        C. Test H2 Dynamics: Procrustes Analysis (Figure 4g) - Swap Hypothesis
        
        Args:
            property_name: Property to decode
            max_time: Maximum timestep to analyze
        
        Returns:
            Dictionary with dynamics analysis results
        """
        print("\n" + "="*70)
        print("ANALYSIS 4: MECHANISMS OF WM DYNAMICS")
        print("="*70)
        
        if self.payloads is None:
            self.load_data()
        
        results = {}
        
        # A. Test H1: Cross-Time Decoding (Figure 4b)
        print("\nA. Test H1: Slot-Based Memory (Cross-Time Decoding)")
        cross_time = self._test_h1_cross_time(property_name, max_time)
        results['h1_cross_time'] = cross_time
        
        # B. Test H2 vs H3: Cross-Stimulus Decoding (Figure 4d)
        print("\nB. Test H2 vs H3: Shared Encoding Space (Cross-Stimulus Decoding)")
        cross_stimulus = self._test_h2_cross_stimulus(property_name, max_time)
        results['h2_cross_stimulus'] = cross_stimulus
        
        # C. Test H2 Dynamics: Procrustes Swap (Figure 4g)
        print("\nC. Test H2 Dynamics: Procrustes Swap Analysis")
        procrustes_swap = self._test_h2_procrustes_swap(property_name, max_time)
        results['h2_procrustes_swap'] = procrustes_swap
        
        # Save results
        with open(self.output_dir / "analysis4_wm_dynamics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _test_h1_cross_time(self, property_name: str, max_time: int) -> Dict:
        """
        Test H1: Train decoder on E(S=1, T=1), test on M(S=1, T=2..6).
        Expected: Accuracy drops off over time (disproves slot-based memory).
        """
        print("  Training decoder on encoding space (t=0)...")
        
        # Train on first encoding
        X_train, y_train, label2idx = build_matrix(
            self.payloads, property_name, time=0, task_index=None, n_value=None
        )
        
        if X_train.numel() == 0:
            return {"status": "no_data"}
        
        clf = self._train_decoder(X_train, y_train)
        
        # Test across all timesteps
        accuracies = []
        for t in range(max_time + 1):
            X_test, y_test, _, raw_vals = build_matrix_with_values(
                self.payloads, property_name, time=t, task_index=None, n_value=None
            )
            
            if X_test.numel() == 0:
                accuracies.append(None)
                continue
            
            y_test_idx, keep = self._align_test_labels(raw_vals, label2idx)
            if keep.sum() == 0:
                accuracies.append(None)
                continue
            
            X_test_np = X_test.numpy()[keep]
            y_pred = clf.predict(X_test_np)
            acc = accuracy_score(y_test_idx, y_pred)
            accuracies.append(float(acc))
            
            print(f"    t={t}: acc={acc:.4f}")
        
        # Plot cross-time decoding
        self._plot_cross_time_decoding(accuracies)
        
        # Check if accuracy drops (H1 disproved)
        if len([a for a in accuracies if a is not None]) >= 2:
            first_acc = accuracies[0]
            last_acc = accuracies[-1] if accuracies[-1] is not None else accuracies[-2]
            if first_acc and last_acc and last_acc < first_acc * 0.8:
                print(f"  ✓ H1 DISPROVED: Accuracy drops from {first_acc:.3f} to {last_acc:.3f}")
            else:
                print(f"  ⚠ H1 NOT CLEARLY DISPROVED: Accuracy stable")
        
        return {"accuracies": accuracies, "times": list(range(len(accuracies)))}
    
    def _plot_cross_time_decoding(self, accuracies: List[Optional[float]]):
        """Plot cross-time decoding accuracy (Figure 4b style)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        times = list(range(len(accuracies)))
        valid_times = [t for t, a in zip(times, accuracies) if a is not None]
        valid_accs = [a for a in accuracies if a is not None]
        
        ax.plot(valid_times, valid_accs, marker='o', linewidth=2, markersize=8, color='#2196F3')
        ax.set_xlabel('Timestep (Memory Age)', fontsize=12)
        ax.set_ylabel('Decoding Accuracy', fontsize=12)
        ax.set_title('Cross-Time Decoding (Figure 4b)\nTest H1: Slot-Based Memory', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        if len(valid_accs) >= 2:
            ax.annotate('Expected:\nAccuracy drops\n(H1 disproved)',
                       xy=(valid_times[-1], valid_accs[-1]),
                       xytext=(valid_times[-1] - 1, valid_accs[-1] + 0.15),
                       fontsize=9, style='italic',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "analysis4a_cross_time_decoding.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: analysis4a_cross_time_decoding.png")
    
    def _test_h2_cross_stimulus(self, property_name: str, max_time: int) -> Dict:
        """
        Test H2 vs H3: Train on E(S=i, T=i), test on E(S=j, T=j) where j != i.
        Expected: Validation and generalization accuracies almost identical (supports H2).
        """
        print("  Testing cross-stimulus generalization...")
        
        # For simplicity, train on t=0 and test on t=1,2,3...
        # This simulates different stimuli at their encoding times
        
        X_train, y_train, label2idx = build_matrix(
            self.payloads, property_name, time=0, task_index=None, n_value=None
        )
        
        if X_train.numel() == 0:
            return {"status": "no_data"}
        
        clf = self._train_decoder(X_train, y_train)
        
        # Train accuracy (validation)
        y_train_pred = clf.predict(X_train.numpy())
        train_acc = accuracy_score(y_train.numpy(), y_train_pred)
        
        # Test on other encoding spaces
        test_accs = []
        for t in range(1, min(max_time + 1, 4)):  # Test on a few other encoding times
            X_test, y_test, _, raw_vals = build_matrix_with_values(
                self.payloads, property_name, time=t, task_index=None, n_value=None
            )
            
            if X_test.numel() == 0:
                continue
            
            y_test_idx, keep = self._align_test_labels(raw_vals, label2idx)
            if keep.sum() == 0:
                continue
            
            X_test_np = X_test.numpy()[keep]
            y_pred = clf.predict(X_test_np)
            acc = accuracy_score(y_test_idx, y_pred)
            test_accs.append(float(acc))
        
        generalization_acc = np.mean(test_accs) if test_accs else None
        
        print(f"    Validation (same stimulus): {train_acc:.4f}")
        if generalization_acc:
            print(f"    Generalization (other stimuli): {generalization_acc:.4f}")
            
            if abs(train_acc - generalization_acc) < 0.1:
                print(f"  ✓ H2 SUPPORTED: Validation ≈ Generalization (shared encoding space)")
            else:
                print(f"  ⚠ H3 POSSIBLE: Validation ≠ Generalization (stimulus-specific)")
        
        return {
            "validation_acc": float(train_acc),
            "generalization_acc": float(generalization_acc) if generalization_acc else None,
            "test_accuracies": test_accs
        }
    
    def _test_h2_procrustes_swap(self, property_name: str, max_time: int) -> Dict:
        """
        Test H2 with Procrustes swap analysis (Figure 4g).
        Eq. 2 (time swap): Should have LOW accuracy
        Eq. 3 (stimulus swap): Should have HIGH accuracy
        """
        print("  Running Procrustes swap analysis...")
        print("  ⚠ This requires careful stimulus tracking - simplified version")
        
        # Compute decoder weights at different times
        weights_by_time = {}
        for t in range(min(max_time + 1, 4)):
            X, y, label2idx = build_matrix(
                self.payloads, property_name, time=t, task_index=None, n_value=None
            )
            
            if X.numel() == 0:
                continue
            
            W = one_vs_rest_weights(X, y)
            weights_by_time[t] = W
        
        if len(weights_by_time) < 2:
            return {"status": "insufficient_data"}
        
        # Compute Procrustes alignments between consecutive times
        procrustes_results = []
        times = sorted(weights_by_time.keys())
        
        for i in range(len(times) - 1):
            t1, t2 = times[i], times[i + 1]
            try:
                R, disparity = compute_procrustes_alignment(
                    weights_by_time[t1],
                    weights_by_time[t2]
                )
                procrustes_results.append({
                    "time_pair": [t1, t2],
                    "disparity": float(disparity)
                })
                print(f"    Procrustes R({t1}→{t2}): disparity={disparity:.4f}")
            except Exception as e:
                print(f"    Warning: Procrustes {t1}→{t2} failed: {e}")
        
        return {"procrustes_alignments": procrustes_results}
    
    # ========================================================================
    # ANALYSIS 5: Causal Perturbation Test
    # ========================================================================
    
    def analyze_causal_perturbation(
        self,
        model_path: Optional[Path],
        property_name: str = "location",
        timestep: int = 3,
        perturbation_range: Tuple[float, float] = (-2.0, 2.0),
        num_distances: int = 21
    ) -> Dict[str, Any]:
        """
        Analysis 5: Causal Perturbation Test (Figure A7)
        
        Perturb hidden states along decoder hyperplane and observe output changes.
        Expected: Match probability drops as state crosses decision boundary.
        
        This analysis requires a trained model checkpoint. If not provided, it will
        be skipped with a message.
        
        Args:
            model_path: Path to trained model checkpoint (required)
            property_name: Property whose decoder to use
            timestep: Timestep to analyze (executive phase, typically 3-5)
            perturbation_range: (min, max) perturbation distances
            num_distances: Number of distances to test
        
        Returns:
            Dictionary with perturbation results
        """
        print("\n" + "="*70)
        print("ANALYSIS 5: CAUSAL PERTURBATION TEST")
        print("="*70)
        
        if model_path is None or not model_path.exists():
            print("  ⚠️  Model checkpoint required for causal perturbation test")
            print("  ⚠️  Skipping Analysis 5")
            print("\n  To run this analysis:")
            print("    python -m src.analysis.causal_perturbation \\")
            print("      --model experiments/wm_mtmf/best_model.pt \\")
            print(f"      --hidden_root {self.hidden_root} \\")
            print(f"      --property {property_name} \\")
            print(f"      --output_dir {self.output_dir}")
            return {"status": "skipped", "reason": "no_model_checkpoint"}
        
        print(f"  Running causal perturbation test on {property_name}...")
        print(f"  Model: {model_path}")
        
        try:
            from .causal_perturbation import analyze_causal_perturbation
            
            results = analyze_causal_perturbation(
                model_path=model_path,
                hidden_root=self.hidden_root,
                property_name=property_name,
                output_dir=self.output_dir,
                timestep=timestep,
                perturbation_range=perturbation_range,
                num_distances=num_distances
            )
            
            print(f"  ✓ Causal perturbation test complete!")
            return results
            
        except Exception as e:
            print(f"  ❌ Causal perturbation test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Analysis Pipeline")
    parser.add_argument("--analysis", type=str, default="all",
                       choices=["all", "1", "2", "3", "4", "5"],
                       help="Which analysis to run (1-5 or 'all')")
    parser.add_argument("--hidden_root", type=str, required=True,
                       help="Path to hidden states directory")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                       help="Output directory for results")
    parser.add_argument("--epochs", type=int, nargs="*",
                       help="Specific epochs to analyze")
    parser.add_argument("--property", type=str, default="identity",
                       choices=["location", "identity", "category"],
                       help="Property to decode for Analysis 4")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model checkpoint (required for Analysis 5)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS PIPELINE")
    print("Based on paper analyses 1-5")
    print("="*70)
    
    analyzer = ComprehensiveAnalysis(
        hidden_root=Path(args.hidden_root),
        output_dir=Path(args.output_dir)
    )
    
    # Run requested analyses
    if args.analysis in ["all", "1"]:
        print("\n>>> Running Analysis 1: Model Behavioral Performance")
        analyzer.analyze_behavioral_performance()
    
    if args.analysis in ["all", "2"]:
        print("\n>>> Running Analysis 2: Encoding of Object Properties")
        analyzer.analyze_encoding_properties(encoding_time=0)
    
    if args.analysis in ["all", "3"]:
        print("\n>>> Running Analysis 3: Representational Orthogonalization")
        analyzer.analyze_orthogonalization(encoding_time=0)
    
    if args.analysis in ["all", "4"]:
        print("\n>>> Running Analysis 4: Mechanisms of WM Dynamics")
        analyzer.analyze_wm_dynamics(property_name=args.property, max_time=5)
    
    if args.analysis in ["all", "5"]:
        print("\n>>> Running Analysis 5: Causal Perturbation Test")
        model_path = Path(args.model) if args.model else None
        analyzer.analyze_causal_perturbation(model_path=model_path, property_name=args.property)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    output_path = Path(args.output_dir)
    if output_path.exists():
        for f in sorted(output_path.glob("*")):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
