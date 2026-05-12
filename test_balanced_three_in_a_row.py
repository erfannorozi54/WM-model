#!/usr/bin/env python3
"""Test balanced three_in_a_row generation."""
import sys
sys.path.insert(0, 'src')

from src.train import load_real_stimulus_data
from src.meta.tasks import generate_three_in_a_row_sequences

stimulus_data = load_real_stimulus_data()

# Generate sequences
sequences = generate_three_in_a_row_sequences(
    stimulus_data=stimulus_data,
    num_sequences=10,
    sequence_length=6,
    task_feature="category"
)

print("Testing 10 sequences for correctness and balance:\n")

total_matches = 0
total_non_matches = 0
total_no_action = 0
errors = 0

for i, seq in enumerate(sequences):
    print(f"Sequence {i}:")
    categories = [t['category'] for t in seq['trials']]
    targets = [t['target'] for t in seq['trials']]
    
    # Count labels
    no_action = targets[:2].count(0)
    matches = targets[2:].count(2)
    non_matches = targets[2:].count(1)
    
    total_no_action += no_action
    total_matches += matches
    total_non_matches += non_matches
    
    print(f"  Categories: {categories}")
    print(f"  Targets:    {targets}")
    print(f"  Labels: No Action={no_action}, Match={matches}, Non-Match={non_matches}")
    
    # Verify first 2 are no_action
    if targets[0] != 0 or targets[1] != 0:
        print(f"  ❌ ERROR: First 2 trials should be no_action!")
        errors += 1
    
    # Verify balance (50-50 for t>=2)
    if matches != non_matches:
        print(f"  ⚠️  WARNING: Not balanced! {matches} matches vs {non_matches} non-matches")
    
    # Verify correctness of labels
    for t in range(len(seq['trials'])):
        expected_label = targets[t]
        
        if t < 2:
            if expected_label != 0:
                print(f"  ❌ ERROR at t={t}: Expected no_action (0), got {expected_label}")
                errors += 1
        else:
            # Check if label matches actual pattern
            if expected_label == 2:  # Match
                if categories[t] != categories[t-1] or categories[t] != categories[t-2]:
                    print(f"  ❌ ERROR at t={t}: Label is Match but pattern is {categories[t-2]}, {categories[t-1]}, {categories[t]}")
                    errors += 1
            elif expected_label == 1:  # Non-match
                if categories[t] == categories[t-1] == categories[t-2]:
                    print(f"  ❌ ERROR at t={t}: Label is Non-Match but pattern is {categories[t-2]}, {categories[t-1]}, {categories[t]} (3 consecutive!)")
                    errors += 1
    
    if errors == 0:
        print(f"  ✓ Correct")
    print()

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Total No Action: {total_no_action} (expected: 20)")
print(f"Total Matches: {total_matches} (expected: 20)")
print(f"Total Non-Matches: {total_non_matches} (expected: 20)")
print(f"Match/Non-Match ratio: {total_matches}:{total_non_matches}")
print(f"Errors: {errors}")

if errors == 0 and total_matches == 20 and total_non_matches == 20:
    print("\n✓ ALL TESTS PASSED!")
else:
    print(f"\n❌ TESTS FAILED!")
