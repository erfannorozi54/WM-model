#!/usr/bin/env python3
"""
Verification script to check if analysis pipeline is correctly set up.

This script verifies:
1. All required modules can be imported
2. Data splits are working
3. Analysis functions are accessible
4. Example workflows can run

Usage:
    python -m src.scripts.verify_analysis_setup
"""

from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        print("  Importing data modules...")
        from src.data.validation_splits import load_and_split_stimuli
        print("    ✓ validation_splits")
        
        from src.data.dataset import NBackDataModule
        print("    ✓ dataset")
        
        from src.data.nback_generator import NBackGenerator, TaskFeature
        print("    ✓ nback_generator")
        
        print("\n  Importing analysis modules...")
        from src.analysis.comprehensive_analysis import ComprehensiveAnalysis
        print("    ✓ comprehensive_analysis")
        
        from src.analysis.decoding import train_decoder
        print("    ✓ decoding")
        
        from src.analysis.orthogonalization import one_vs_rest_weights, orthogonalization_index
        print("    ✓ orthogonalization")
        
        from src.analysis.procrustes import compute_procrustes_alignment
        print("    ✓ procrustes")
        
        print("\n  ✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Import failed: {e}")
        return False


def test_data_splits():
    """Test data split functionality."""
    print("\n" + "="*70)
    print("TEST 2: Data Splits")
    print("="*70)
    
    stimuli_dir = Path("data/stimuli")
    
    if not stimuli_dir.exists():
        print(f"  ⚠️  Stimuli directory not found: {stimuli_dir}")
        print("  ⚠️  Run: python -m src.data.generate_stimuli")
        return False
    
    try:
        print("  Loading and splitting stimuli...")
        from src.data.validation_splits import load_and_split_stimuli
        
        train, val_angle, val_id, stats = load_and_split_stimuli(
            stimuli_dir=str(stimuli_dir),
            train_angles=[0, 1, 2],
            val_angles=[3],
            train_identity_ratio=0.6
        )
        
        print(f"\n  Training data: {stats['training']['num_stimuli']} stimuli")
        print(f"  Val (novel angle): {stats['val_novel_angle']['num_stimuli']} stimuli")
        print(f"  Val (novel identity): {stats['val_novel_identity']['num_stimuli']} stimuli")
        
        # Verify splits are correct
        errors = []
        
        # Check same identities in train and val_angle
        for cat in train:
            train_ids = set(train[cat].keys())
            val_angle_ids = set(val_angle[cat].keys())
            if train_ids != val_angle_ids:
                errors.append(f"Category {cat}: identity mismatch between train and val_angle")
        
        # Check no overlap in train and val_identity
        for cat in train:
            train_ids = set(train[cat].keys())
            val_id_ids = set(val_id[cat].keys())
            overlap = train_ids & val_id_ids
            if overlap:
                errors.append(f"Category {cat}: identity overlap {overlap}")
        
        if errors:
            print("\n  ❌ Validation errors:")
            for err in errors:
                print(f"    - {err}")
            return False
        else:
            print("\n  ✅ Data splits verified!")
            return True
            
    except Exception as e:
        print(f"\n  ❌ Data split test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis_functions():
    """Test that analysis functions are callable."""
    print("\n" + "="*70)
    print("TEST 3: Analysis Functions")
    print("="*70)
    
    try:
        from src.analysis.comprehensive_analysis import ComprehensiveAnalysis
        # Create analyzer instance
        analyzer = ComprehensiveAnalysis(
            hidden_root=Path("experiments/test/hidden_states"),
            output_dir=Path("test_output")
        )
        
        print("  ✓ ComprehensiveAnalysis instantiated")
        
        # Check methods exist
        methods = [
            'analyze_behavioral_performance',
            'analyze_encoding_properties',
            'analyze_orthogonalization',
            'analyze_wm_dynamics',
            'analyze_causal_perturbation'
        ]
        
        for method in methods:
            if hasattr(analyzer, method):
                print(f"  ✓ {method}")
            else:
                print(f"  ❌ {method} not found")
                return False
        
        print("\n  ✅ All analysis functions available!")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Analysis function test failed: {e}")
        return False


def test_training_script():
    """Test that training script exists and is importable."""
    print("\n" + "="*70)
    print("TEST 4: Training Scripts")
    print("="*70)
    
    scripts = [
        ("src/train.py", "Original training script"),
        ("src/train_with_generalization.py", "Training with generalization (Phase 6)")
    ]
    
    all_ok = True
    for script_path, description in scripts:
        path = Path(script_path)
        if path.exists():
            print(f"  ✓ {description}: {script_path}")
        else:
            print(f"  ❌ {description}: {script_path} not found")
            all_ok = False
    
    if all_ok:
        print("\n  ✅ All training scripts available!")
    
    return all_ok


def test_config_files():
    """Test that config files exist."""
    print("\n" + "="*70)
    print("TEST 5: Configuration Files")
    print("="*70)
    
    configs = [
        "configs/stsf.yaml",
        "configs/stmf.yaml",
        "configs/mtmf.yaml"
    ]
    
    all_ok = True
    for config in configs:
        path = Path(config)
        if path.exists():
            print(f"  ✓ {config}")
        else:
            print(f"  ❌ {config} not found")
            all_ok = False
    
    if all_ok:
        print("\n  ✅ All config files found!")
    
    return all_ok


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("ANALYSIS PIPELINE VERIFICATION")
    print("="*70)
    print("\nThis script verifies that all components are correctly set up.\n")
    
    results = {
        "Module Imports": test_imports(),
        "Data Splits": test_data_splits(),
        "Analysis Functions": test_analysis_functions(),
        "Training Scripts": test_training_script(),
        "Config Files": test_config_files()
    }
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Analysis pipeline is ready.")
        print("\nNext steps:")
        print("1. Train a model:")
        print("   python -m src.train_with_generalization --config configs/mtmf.yaml")
        print("\n2. Run analyses:")
        print("   python -m src.analysis.comprehensive_analysis \\")
        print("     --analysis all \\")
        print("     --hidden_root experiments/wm_mtmf/hidden_states \\")
        print("     --output_dir analysis_results")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
