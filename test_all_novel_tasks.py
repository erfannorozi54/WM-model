#!/usr/bin/env python3
"""Test all novel task generation to ensure intentional matches."""
import sys
sys.path.insert(0, 'src')

from src.train import load_real_stimulus_data
from src.meta.tasks import generate_novel_sequences, NOVEL_TASKS

stimulus_data = load_real_stimulus_data()

for task_name in NOVEL_TASKS.keys():
    print(f"\n{'='*70}")
    print(f"Testing: {task_name}")
    print(f"{'='*70}")
    
    # Generate sequences
    sequences = generate_novel_sequences(
        task_name=task_name,
        stimulus_data=stimulus_data,
        num_sequences=3,
        task_feature="category",
        sequence_length=6
    )
    
    for i, seq in enumerate(sequences):
        print(f"\nSequence {i}:")
        targets = [t['target'] if isinstance(t, dict) else t.target_response for t in seq['trials']]
        
        # Count labels
        no_action = targets.count(0)
        non_match = targets.count(1)
        match = targets.count(2)
        
        print(f"  Labels: No Action={no_action}, Non-Match={non_match}, Match={match}")
        
        # Check if there are matches
        if match == 0:
            print(f"  ⚠️  WARNING: No matches in sequence!")
        else:
            print(f"  ✓ Has {match} match(es)")
        
        # Show trial details
        for t_idx, trial in enumerate(seq['trials']):
            if isinstance(trial, dict):
                target = trial['target']
                cat = trial['category']
            else:
                target = trial.target_response
                cat = trial.category
            
            label_name = ["No Action", "Non-Match", "Match"][target]
            print(f"    t={t_idx}: {cat} → {label_name}")
