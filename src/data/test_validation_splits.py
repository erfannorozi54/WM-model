"""
Test script for validation data splits.
Verifies that training, novel-angle, and novel-identity splits are correctly separated.
"""

from .validation_splits import load_and_split_stimuli
import json


def test_validation_splits():
    """Test the validation split functionality."""
    
    print("="*70)
    print("TESTING VALIDATION DATA SPLITS")
    print("="*70)
    
    # Load and split stimuli
    print("\n1. Loading and splitting stimuli...")
    train_data, val_novel_angle_data, val_novel_identity_data, stats = load_and_split_stimuli(
        stimuli_dir="data/stimuli",
        train_angles=[0, 1, 2],  # Use angles 0, 1, 2 for training
        val_angles=[3],           # Use angle 3 for novel-angle validation
        train_identity_ratio=0.6  # 60% for training, 40% for validation
    )
    
    print("\n2. Split Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Verify splits
    print("\n3. Verifying Splits...")
    
    # Check that training and novel-angle validation have same identities
    print("\n   a) Checking Novel-Angle Validation (same identities, different angles):")
    for category in train_data:
        train_ids = set(train_data[category].keys())
        val_angle_ids = set(val_novel_angle_data[category].keys())
        
        if train_ids == val_angle_ids:
            print(f"      ✓ {category}: Same identities in training and novel-angle val")
        else:
            print(f"      ✗ {category}: Identity mismatch!")
            print(f"        Train: {train_ids}")
            print(f"        Val:   {val_angle_ids}")
    
    # Check that novel-identity validation has different identities
    print("\n   b) Checking Novel-Identity Validation (different identities):")
    for category in train_data:
        train_ids = set(train_data[category].keys())
        val_id_ids = set(val_novel_identity_data[category].keys())
        
        overlap = train_ids & val_id_ids
        if len(overlap) == 0:
            print(f"      ✓ {category}: No overlap between training and novel-identity val")
        else:
            print(f"      ✗ {category}: Found overlap: {overlap}")
    
    # Sample and inspect some files
    print("\n4. Sample Files:")
    for category in list(train_data.keys())[:1]:  # Just show one category
        train_ids = list(train_data[category].keys())
        if train_ids:
            sample_id = train_ids[0]
            print(f"\n   Training ({category}, {sample_id}):")
            for path in train_data[category][sample_id][:3]:
                print(f"     - {path}")
            
            print(f"\n   Novel-Angle Validation ({category}, {sample_id}):")
            for path in val_novel_angle_data[category][sample_id][:3]:
                print(f"     - {path}")
        
        val_ids = list(val_novel_identity_data[category].keys())
        if val_ids:
            sample_id = val_ids[0]
            print(f"\n   Novel-Identity Validation ({category}, {sample_id}):")
            for path in val_novel_identity_data[category][sample_id][:3]:
                print(f"     - {path}")
    
    print("\n" + "="*70)
    print("✓ VALIDATION SPLITS TEST COMPLETED")
    print("="*70)
    
    return train_data, val_novel_angle_data, val_novel_identity_data, stats


if __name__ == "__main__":
    test_validation_splits()
