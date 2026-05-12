#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from src.train import load_real_stimulus_data
from src.meta.tasks import generate_three_in_a_row_sequences

stimulus_data = load_real_stimulus_data()

# Generate sequences
sequences = generate_three_in_a_row_sequences(
    stimulus_data=stimulus_data,
    num_sequences=5,
    sequence_length=6,
    task_feature="category"
)

for i, seq in enumerate(sequences):
    print(f"\n=== Sequence {i} ===")
    categories = [t['category'] for t in seq['trials']]
    targets = [t['target'] for t in seq['trials']]
    
    print(f"Categories: {categories}")
    print(f"Targets:    {targets}")
    
    # Verify labeling
    for t in range(len(seq['trials'])):
        if t < 2:
            expected = 0
        elif categories[t] == categories[t-1] == categories[t-2]:
            expected = 2
        else:
            expected = 0
        
        actual = targets[t]
        status = "✓" if actual == expected else "✗"
        print(f"  t={t}: {categories[t]} → target={actual} (expected={expected}) {status}")
