#!/usr/bin/env python3
"""
Demo with real generated stimulus images.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.dataset import NBackDataModule
from data.nback_generator import TaskFeature


def load_real_stimulus_data():
    """Load stimulus data from generated files."""
    stimuli_dir = Path("data/stimuli")
    
    stimulus_data = {}
    stimulus_files = list(stimuli_dir.glob("stimulus_*.png"))
    
    for file_path in stimulus_files:
        parts = file_path.stem.split('_')
        if len(parts) >= 4:
            category = parts[1]
            identity = f"{parts[1]}_{parts[2]}"
            
            if category not in stimulus_data:
                stimulus_data[category] = {}
            if identity not in stimulus_data[category]:
                stimulus_data[category][identity] = []
                
            stimulus_data[category][identity].append(str(file_path))
    
    # Sort for consistency
    for category in stimulus_data:
        for identity in stimulus_data[category]:
            stimulus_data[category][identity].sort()
    
    return stimulus_data


def main():
    print("Demo with Real Stimulus Images")
    print("=" * 40)
    
    # Load real stimulus data
    stimulus_data = load_real_stimulus_data()
    
    if not stimulus_data:
        print("No stimulus images found. Run 'python generate_stimuli.py' first.")
        return
    
    print("Available stimulus data:")
    for category, identities in stimulus_data.items():
        print(f"  {category}: {len(identities)} identities")
    
    # Create data module with real stimuli
    data_module = NBackDataModule(
        stimulus_data=stimulus_data,
        n_values=[1, 2],
        task_features=[TaskFeature.LOCATION, TaskFeature.CATEGORY],
        sequence_length=6,
        batch_size=2,
        num_train=10,
        num_val=5,
        num_test=5,
        num_workers=0
    )
    
    # Sample a batch
    print("\nSampling batch with real images...")
    try:
        batch = data_module.sample_batch("train")
        print(f"✓ Successfully loaded batch with real images!")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  No file loading warnings!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
