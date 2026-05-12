"""Novel task definitions and sequence generators for meta-learning experiments."""

from typing import Dict, List
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
    
    The task is to detect when the same feature value appears 3+ consecutive times.
    Matches start at the 3rd consecutive item and continue until the pattern breaks.
    
    Example: A, A, A (match), A (match), B, B, B (match)
             t=0,1,   t=2,       t=3,      t=4,5,   t=6
    """
    sequences = []
    categories = list(stimulus_data.keys())
    
    for _ in range(num_sequences):
        trials = []
        feature_values = []  # Track actual feature values
        
        # Pre-generate a pattern with intentional three-in-a-row sequences
        pattern = []
        t = 0
        while t < sequence_length:
            # 50% chance to create a three-in-a-row pattern
            if torch.rand(1).item() > 0.5 and t + 3 <= sequence_length:
                # Create 3-4 consecutive same values
                length = min(torch.randint(3, 5, (1,)).item(), sequence_length - t)
                value = torch.randint(0, len(categories), (1,)).item()
                pattern.extend([value] * length)
                t += length
            else:
                # Random value
                pattern.append(torch.randint(0, len(categories), (1,)).item())
                t += 1
        
        pattern = pattern[:sequence_length]  # Trim to exact length
        
        # Generate trials based on the pattern
        for t in range(sequence_length):
            cat_idx = pattern[t]
            category = categories[cat_idx]
            identities = list(stimulus_data[category].keys())
            
            # For location and identity, we need to control them based on task_feature
            if task_feature == "location":
                # Control location to match pattern
                location = pattern[t] % 4  # Map to 4 locations
                ident = torch.randint(0, len(identities), (1,)).item()
                identity = identities[ident]
                current_value = location
            elif task_feature == "identity":
                # Control identity to match pattern
                ident = pattern[t] % len(identities)
                identity = identities[ident]
                location = torch.randint(0, 4, (1,)).item()
                current_value = identity
            else:  # category
                # Category already controlled by pattern
                ident = torch.randint(0, len(identities), (1,)).item()
                identity = identities[ident]
                location = torch.randint(0, 4, (1,)).item()
                current_value = category
            
            feature_values.append(current_value)
            
            stimuli = stimulus_data[category][identity]
            stim_idx = torch.randint(0, len(stimuli), (1,)).item()
            stimulus_path = stimuli[stim_idx]
            
            # Determine target based on actual feature values
            if t < 2:
                target = 0  # no_action for first 2 trials
            elif feature_values[t] == feature_values[t-1] == feature_values[t-2]:
                target = 2  # match (3+ in a row)
            else:
                target = 0  # no_action (pattern broken)
            
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
    
    for _ in range(num_sequences):
        trials = []
        location_history = []
        identity_history = []
        
        for t in range(sequence_length):
            cat = torch.randint(0, len(categories), (1,)).item()
            category = categories[cat]
            identities = list(stimulus_data[category].keys())
            ident = torch.randint(0, len(identities), (1,)).item()
            identity = identities[ident]
            stimuli = stimulus_data[category][identity]
            stim_idx = torch.randint(0, len(stimuli), (1,)).item()
            location = torch.randint(0, 4, (1,)).item()
            
            stimulus_path = stimuli[stim_idx]
            location_history.append(location)
            identity_history.append(identity)
            
            # Alternate: even timesteps check location, odd timesteps check identity
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
    """Generate standard N-back sequences for a specific N value."""
    generator = NBackGenerator(
        stimulus_data=stimulus_data,
        n_locations=4,
        sequence_length=sequence_length,
        match_probability=0.3,
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
