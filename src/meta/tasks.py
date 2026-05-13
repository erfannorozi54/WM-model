"""Novel task definitions and sequence generators for meta-learning experiments."""

from typing import Dict, List
import random
import torch

from ..data.nback_generator import NBackGenerator, TaskFeature


NOVEL_TASKS = {
    "nback_4": {
        "description": "4-back task (N=4, not seen during training)",
        "n_value": 4,
        "task_type": "standard",
        "features": ["location", "identity", "category"],
    },
    "nback_5": {
        "description": "5-back task (N=5, tests capacity limits)",
        "n_value": 5,
        "task_type": "standard",
        "features": ["location", "identity", "category"],
    },
    "three_in_a_row": {
        "description": "Detect when same feature appears 3 consecutive times",
        "n_value": 0,
        "task_type": "pattern",
        "features": ["location", "identity", "category"],
    },
    "alternating": {
        "description": "Alternate between two features each timestep",
        "n_value": 2,
        "task_type": "alternating",
        "features": ["location", "identity"],
    },
}


def generate_three_in_a_row_sequences(
    stimulus_data: Dict,
    num_sequences: int,
    sequence_length: int = 6,
    task_feature: str = "category",
) -> List[Dict]:
    """Generate sequences for 'three-in-a-row' pattern detection task.
    
    Strategy: Pre-allocate match counts with bell curve distribution.
    - 1% sequences with 0 matches
    - 22% sequences with 1 match
    - 54% sequences with 2 matches (center)
    - 22% sequences with 3 matches
    - 1% sequences with 4 matches
    
    This creates a natural, symmetric bell curve centered at 2 matches,
    while maintaining exact 50% overall balance.
    """
    sequences = []
    categories = list(stimulus_data.keys())
    num_values = len(categories)
    
    # Pre-allocate match counts with bell curve distribution (symmetric around 2)
    # Distribution: 0:1%, 1:22%, 2:54%, 3:22%, 4:1%
    num_actionable = sequence_length - 2  # 4 actionable trials
    distribution = [0.01, 0.22, 0.54, 0.22, 0.01]  # Steeper bell curve centered at 2
    
    match_counts = []
    for target_matches in range(num_actionable + 1):  # 0, 1, 2, 3, 4
        count = int(num_sequences * distribution[target_matches])
        match_counts.extend([target_matches] * count)
    
    # Handle remainder (add to center)
    remainder = num_sequences - len(match_counts)
    match_counts.extend([2] * remainder)
    
    # Shuffle to randomize order
    random.shuffle(match_counts)
    
    for target_matches in match_counts:
        # Deterministically construct pattern with exact number of matches
        if target_matches == 0:
            # No matches: ensure no 3 consecutive same values
            pattern = [torch.randint(0, num_values, (1,)).item() for _ in range(sequence_length)]
            for t in range(2, sequence_length):
                while pattern[t] == pattern[t-1] == pattern[t-2]:
                    pattern[t] = (pattern[t] + 1) % num_values
        
        elif target_matches == 4:
            # All 4 actionable trials match: need pattern like [A, B, X, X, X, X]
            val = torch.randint(0, num_values, (1,)).item()
            pattern = [
                torch.randint(0, num_values, (1,)).item(),
                torch.randint(0, num_values, (1,)).item(),
                val, val, val, val
            ]
        
        else:
            # For 1, 2, or 3 matches: use retry with early termination
            max_attempts = 100
            for attempt in range(max_attempts):
                pattern = [torch.randint(0, num_values, (1,)).item() for _ in range(sequence_length)]
                
                # Count matches
                matches = 0
                for t in range(2, sequence_length):
                    if pattern[t] == pattern[t-1] == pattern[t-2]:
                        matches += 1
                
                if matches == target_matches:
                    break
            
            # If failed, construct deterministically
            if matches != target_matches:
                pattern = [torch.randint(0, num_values, (1,)).item() for _ in range(2)]
                remaining = sequence_length - 2
                
                # Add target_matches matching positions
                for _ in range(target_matches):
                    val = pattern[-1] if len(pattern) >= 2 else torch.randint(0, num_values, (1,)).item()
                    pattern.append(val)
                
                # Fill rest with non-matching values
                for _ in range(remaining - target_matches):
                    val = torch.randint(0, num_values, (1,)).item()
                    while len(pattern) >= 2 and val == pattern[-1] == pattern[-2]:
                        val = (val + 1) % num_values
                    pattern.append(val)
        
        # Calculate labels
        labels = []
        for t in range(sequence_length):
            if t < 2:
                labels.append(0)  # no_action
            elif pattern[t] == pattern[t-1] == pattern[t-2]:
                labels.append(2)  # match
            else:
                labels.append(1)  # non_match
        
        # Generate trials from pattern
        trials = []
        for t in range(sequence_length):
            cat_idx = pattern[t]
            category = categories[cat_idx]
            identities = list(stimulus_data[category].keys())
            
            if task_feature == "location":
                location = pattern[t] % 4
                ident = torch.randint(0, len(identities), (1,)).item()
                identity = identities[ident]
            elif task_feature == "identity":
                ident = pattern[t] % len(identities)
                identity = identities[ident]
                location = torch.randint(0, 4, (1,)).item()
            else:  # category
                ident = torch.randint(0, len(identities), (1,)).item()
                identity = identities[ident]
                location = torch.randint(0, 4, (1,)).item()
            
            stimuli = stimulus_data[category][identity]
            stim_idx = torch.randint(0, len(stimuli), (1,)).item()
            stimulus_path = stimuli[stim_idx]
            
            trials.append({
                "stimulus_path": stimulus_path,
                "location": location,
                "category": category,
                "identity": identity,
                "target": labels[t],
                "trial_index": t,
            })
        
        sequences.append({
            "trials": trials,
            "task_vector": torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            "n": 0,
            "task_feature": task_feature,
        })
    
    return sequences


def generate_alternating_sequences(
    stimulus_data: Dict,
    num_sequences: int,
    sequence_length: int = 6,
    n_value: int = 2,
) -> List[Dict]:
    """Generate sequences for alternating feature task (location/identity switch each timestep).
    
    Even timesteps check location N-back, odd timesteps check identity N-back.
    """
    sequences = []
    categories = list(stimulus_data.keys())
    match_probability = 0.3
    
    for _ in range(num_sequences):
        trials = []
        location_history = []
        identity_history = []
        
        for t in range(sequence_length):
            # Determine if this should be a match trial
            should_match = t >= n_value and torch.rand(1).item() < match_probability
            
            # Select category
            cat = torch.randint(0, len(categories), (1,)).item()
            category = categories[cat]
            identities = list(stimulus_data[category].keys())
            
            # Determine location and identity based on whether it should match
            if should_match:
                if t % 2 == 0:
                    # Even timestep: match location
                    location = location_history[t - n_value]
                    ident = torch.randint(0, len(identities), (1,)).item()
                    identity = identities[ident]
                else:
                    # Odd timestep: match identity
                    identity = identity_history[t - n_value]
                    location = torch.randint(0, 4, (1,)).item()
            else:
                # Random selection
                ident = torch.randint(0, len(identities), (1,)).item()
                identity = identities[ident]
                location = torch.randint(0, 4, (1,)).item()
            
            stimuli = stimulus_data[category][identity]
            stim_idx = torch.randint(0, len(stimuli), (1,)).item()
            stimulus_path = stimuli[stim_idx]
            
            location_history.append(location)
            identity_history.append(identity)
            
            # Determine target based on actual values
            if t < n_value:
                target = 0  # no_action for first N trials
            elif t % 2 == 0:
                # Check location N-back
                if location == location_history[t - n_value]:
                    target = 2  # match
                else:
                    target = 1  # non_match
            else:
                # Check identity N-back
                if identity == identity_history[t - n_value]:
                    target = 2  # match
                else:
                    target = 1  # non_match
            
            trials.append({
                "stimulus_path": stimulus_path,
                "location": location,
                "category": category,
                "identity": identity,
                "target": target,
                "trial_index": t,
            })
        
        sequences.append({
            "trials": trials,
            "task_vector": torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            "n": n_value,
            "task_feature": "alternating",
        })
    
    return sequences


def generate_standard_nback_sequences(
    stimulus_data: Dict,
    n_value: int,
    task_feature: str,
    num_sequences: int,
    sequence_length: int = 6,
) -> List[Dict]:
    """Generate standard N-back sequences for a specific N value with balanced labels."""
    generator = NBackGenerator(
        stimulus_data=stimulus_data,
        n_locations=4,
        sequence_length=sequence_length,
        match_probability=0.5,  # Changed from 0.3 to 0.5 for balanced labels
    )
    
    feature_enum = {
        "location": TaskFeature.LOCATION,
        "identity": TaskFeature.IDENTITY,
        "category": TaskFeature.CATEGORY,
    }
    
    sequences = []
    for _ in range(num_sequences):
        seq = generator.generate_sequence(n=n_value, task_feature=feature_enum[task_feature])
        sequences.append({
            "trials": seq.trials,
            "task_vector": seq.task_vector,
            "n": n_value,
            "task_feature": task_feature,
        })
    
    return sequences


def generate_novel_sequences(
    task_name: str,
    stimulus_data: Dict,
    num_sequences: int,
    task_feature: str = "category",
    sequence_length: int = 6,
) -> List[Dict]:
    """Generate sequences for a novel task."""
    if task_name not in NOVEL_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(NOVEL_TASKS.keys())}")
    
    task_config = NOVEL_TASKS[task_name]
    
    if task_config["task_type"] == "standard":
        return generate_standard_nback_sequences(
            stimulus_data=stimulus_data,
            n_value=task_config["n_value"],
            task_feature=task_feature,
            num_sequences=num_sequences,
            sequence_length=sequence_length,
        )
    elif task_config["task_type"] == "pattern":
        return generate_three_in_a_row_sequences(
            stimulus_data=stimulus_data,
            num_sequences=num_sequences,
            sequence_length=sequence_length,
            task_feature=task_feature,
        )
    elif task_config["task_type"] == "alternating":
        return generate_alternating_sequences(
            stimulus_data=stimulus_data,
            num_sequences=num_sequences,
            sequence_length=sequence_length,
            n_value=task_config["n_value"],
        )
    else:
        raise ValueError(f"Unknown task type: {task_config['task_type']}")
