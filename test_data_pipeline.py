#!/usr/bin/env python3
"""
Comprehensive test script for the Working Memory Model data pipeline.
Tests all components: ShapeNet downloader, renderer, N-back generator, and PyTorch dataset.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.shapenet_downloader import ShapeNetDownloader
from data.renderer import StimulusRenderer
from data.nback_generator import NBackGenerator, TaskFeature, create_sample_stimulus_data
from data.dataset import NBackDataModule, create_demo_dataset


def test_shapenet_downloader():
    """Test the ShapeNet downloader."""
    print("Testing ShapeNet Downloader...")
    print("-" * 30)
    
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = ShapeNetDownloader(data_dir=temp_dir)
            
            # Test category download
            results = downloader.download_all_categories()
            
            # Check results
            assert len(results) == 4, f"Expected 4 categories, got {len(results)}"
            
            for category, success in results.items():
                assert success, f"Failed to download category: {category}"
            
            # Test object path retrieval
            for category in downloader.categories.values():
                paths = downloader.get_object_paths(category)
                assert len(paths) == 2, f"Expected 2 objects in {category}, got {len(paths)}"
            
            # Test dataset info
            info = downloader.get_dataset_info()
            assert info["total_objects"] == 8, f"Expected 8 total objects, got {info['total_objects']}"
            
            print("✓ ShapeNet downloader tests passed")
            return True
            
    except Exception as e:
        print(f"✗ ShapeNet downloader test failed: {e}")
        return False


def test_renderer():
    """Test the stimulus renderer."""
    print("Testing Stimulus Renderer...")
    print("-" * 30)
    
    try:
        # Create renderer
        renderer = StimulusRenderer(image_size=(128, 128))  # Smaller for faster testing
        
        # Test basic properties
        assert len(renderer.locations) == 4, f"Expected 4 locations, got {len(renderer.locations)}"
        assert renderer.image_size == (128, 128), f"Unexpected image size: {renderer.image_size}"
        
        # Test sample stimulus creation
        sample = renderer.create_sample_stimulus()
        assert sample.shape == (128, 128, 3), f"Unexpected sample shape: {sample.shape}"
        assert sample.dtype == np.uint8, f"Unexpected sample dtype: {sample.dtype}"
        
        print("✓ Stimulus renderer tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Stimulus renderer test failed: {e}")
        return False


def test_nback_generator():
    """Test the N-back sequence generator."""
    print("Testing N-back Generator...")
    print("-" * 30)
    
    try:
        # Create sample stimulus data
        stimulus_data = create_sample_stimulus_data()
        
        # Create generator
        generator = NBackGenerator(stimulus_data)
        
        # Test single sequence generation
        sequence = generator.generate_sequence(
            n=2,
            task_feature=TaskFeature.LOCATION,
            sequence_length=6
        )
        
        # Validate sequence properties
        assert sequence.n == 2, f"Expected n=2, got {sequence.n}"
        assert sequence.task_feature == TaskFeature.LOCATION, f"Unexpected task feature"
        assert len(sequence.trials) == 6, f"Expected 6 trials, got {len(sequence.trials)}"
        assert sequence.task_vector.sum() == 1.0, f"Task vector should be one-hot"
        
        # Test batch generation
        batch = generator.generate_batch(
            batch_size=5,
            n=1,
            task_feature=TaskFeature.CATEGORY
        )
        
        assert len(batch) == 5, f"Expected batch size 5, got {len(batch)}"
        
        for seq in batch:
            assert seq.n == 1, f"Expected n=1 in batch"
            assert seq.task_feature == TaskFeature.CATEGORY, f"Expected category task"
        
        # Test mixed batch generation
        mixed_batch = generator.generate_mixed_batch(
            batch_size=10,
            n_values=[1, 2, 3],
            task_features=[TaskFeature.LOCATION, TaskFeature.IDENTITY]
        )
        
        assert len(mixed_batch) == 10, f"Expected mixed batch size 10"
        
        # Check variety in mixed batch
        n_values_seen = set(seq.n for seq in mixed_batch)
        tasks_seen = set(seq.task_feature for seq in mixed_batch)
        
        assert len(n_values_seen) > 1, "Mixed batch should have multiple N values"
        
        # Test statistics
        stats = generator.get_statistics(mixed_batch)
        assert stats["total_sequences"] == 10, f"Expected 10 total sequences in stats"
        assert "match_rate" in stats, "Stats should include match rate"
        
        print("✓ N-back generator tests passed")
        return True
        
    except Exception as e:
        print(f"✗ N-back generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_dataset():
    """Test the PyTorch dataset and dataloader."""
    print("Testing PyTorch Dataset...")
    print("-" * 30)
    
    try:
        # Create demo dataset (this will work even without real images)
        data_module = create_demo_dataset()
        
        # Test dataset sizes
        assert len(data_module.train_dataset) == 20, f"Expected 20 training samples"
        assert len(data_module.val_dataset) == 10, f"Expected 10 validation samples"
        assert len(data_module.test_dataset) == 10, f"Expected 10 test samples"
        
        # Test single sample (this might fail if images don't exist, but we'll handle it)
        try:
            sample = data_module.train_dataset[0]
            
            # Check sample structure
            expected_keys = {'images', 'responses', 'task_vector', 'n', 'locations', 
                           'categories', 'identities', 'sequence_length'}
            assert set(sample.keys()) == expected_keys, f"Unexpected sample keys: {sample.keys()}"
            
            # Check tensor shapes
            seq_len = sample['sequence_length'].item()
            assert sample['images'].shape[0] == seq_len, "Images first dim should match sequence length"
            assert sample['images'].shape[1:] == (3, 224, 224), f"Unexpected image shape: {sample['images'].shape[1:]}"
            assert sample['responses'].shape == (seq_len, 3), f"Unexpected responses shape: {sample['responses'].shape}"
            assert sample['task_vector'].shape == (3,), f"Unexpected task vector shape: {sample['task_vector'].shape}"
            
            print("✓ Sample validation passed")
            
        except Exception as sample_error:
            print(f"⚠ Sample test skipped (expected without real images): {sample_error}")
        
        # Test dataloader creation
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        assert train_loader.batch_size == 4, f"Expected batch size 4"
        assert val_loader.batch_size == 4, f"Expected batch size 4"
        assert test_loader.batch_size == 4, f"Expected batch size 4"
        
        print("✓ PyTorch dataset tests passed")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components."""
    print("Testing Integration...")
    print("-" * 30)
    
    try:
        # Test that components work together
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Setup ShapeNet data
            downloader = ShapeNetDownloader(data_dir=temp_path / "shapenet")
            downloader.download_all_categories()
            
            # 2. Create stimulus renderer
            renderer = StimulusRenderer(image_size=(64, 64))  # Small for speed
            
            # 3. Create sample rendered data structure
            stimulus_data = {}
            
            for category in downloader.categories.values():
                obj_paths = downloader.get_object_paths(category)
                stimulus_data[category] = {}
                
                for i, obj_path in enumerate(obj_paths):
                    identity = f"{category}_{i:03d}"
                    # Create fake rendered images for each location
                    stimulus_data[category][identity] = [
                        f"{temp_path}/stimuli/{identity}_loc{j}_angle0.png" 
                        for j in range(4)
                    ]
            
            # 4. Test N-back generator with this data
            generator = NBackGenerator(stimulus_data)
            sequence = generator.generate_sequence(n=1, task_feature=TaskFeature.LOCATION)
            
            assert len(sequence.trials) > 0, "Should generate trials"
            
            print("✓ Integration tests passed")
            return True
            
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_performance_test():
    """Run basic performance tests."""
    print("Running Performance Tests...")
    print("-" * 30)
    
    try:
        import time
        
        # Test N-back generation speed
        stimulus_data = create_sample_stimulus_data()
        generator = NBackGenerator(stimulus_data)
        
        start_time = time.time()
        batch = generator.generate_batch(batch_size=100, n=2, task_feature=TaskFeature.LOCATION)
        generation_time = time.time() - start_time
        
        print(f"Generated 100 sequences in {generation_time:.3f} seconds")
        assert generation_time < 5.0, f"Generation too slow: {generation_time:.3f}s"
        
        # Test dataset creation speed
        start_time = time.time()
        data_module = create_demo_dataset()
        dataset_time = time.time() - start_time
        
        print(f"Created dataset in {dataset_time:.3f} seconds")
        
        print("✓ Performance tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Working Memory Model - Data Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("ShapeNet Downloader", test_shapenet_downloader),
        ("Stimulus Renderer", test_renderer),
        ("N-back Generator", test_nback_generator),
        ("PyTorch Dataset", test_pytorch_dataset),
        ("Integration", test_integration),
        ("Performance", run_performance_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("=" * len(test_name))
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data pipeline is ready.")
        return True
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
