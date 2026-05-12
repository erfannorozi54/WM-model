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
    
    Generates balanced labels: exactly 50% match, 50% non-match for trials t≥2.
    First 2 trials are always no_action.
    """
    sequences = []
    categories = list(stimulus_data.keys())
    num_values = len(categories)
    
    for _ in range(num_sequences):
        # Step 1: Pre-determine balanced labels
        num_actionable = sequence_length - 2  # Trials after t=0,1
        num_matches = num_actionable // 2
        num_non_matches = num_actionable - num_matches
        
        # Create and shuffle actionable labels
        actionable_labels = [2] * num_matches + [1] * num_non_matches
        indices = torch.randperm(len(actionable_labels))
        actionable_labels = [actionable_labels[i] for i in indices]
        
        labels = [0, 0] + actionable_labels
        
        # Step 2: Generate pattern that satisfies the labels
        pattern = []
        
        # Initialize first two positions randomly
        pattern.append(torch.randint(0, num_values, (1,)).item())
        pattern.append(torch.randint(0, num_values, (1,)).item())
        
        # Generate remaining positions based on labels
        for t in range(2, sequence_length):
            if labels[t] == 2:  # Match needed
                # Current must equal both t-1 and t-2
                # So we need pattern[t-2] == pattern[t-1] == pattern[t]
                # Set t-1 to equal t-2, then set t to equal t-1
                pattern[t-1] = pattern[t-2]
                pattern.append(pattern[t-2])
            else:  # Non-match needed (label == 1)
                # Current must NOT form 3 consecutive
                # Pick value different from t-1
                available = [v for v in range(num_values) if v != pattern[t-1]]
                if available:
                    pattern.append(available[torch.randint(0, len(available), (1,)).item()])
                else:
                    # Fallback (shouldn't happen with 4 categories)
                    pattern.append(pattern[t-1])
        
        # Step 3: Generate trials from pattern
        trials = []
        for t in range(sequence_length):
            cat_idx = pattern[t]
            category = categories[cat_idx]
            identities = list(stimulus_data[category].keys())
            
            # Control feature based on task_feature
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
