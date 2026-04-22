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
    sequence_length: int = 8,
    task_feature: str = "category",
) -> List[Dict]:
    """Generate sequences for 'three-in-a-row' pattern detection task."""
    sequences = []
    categories = list(stimulus_data.keys())
    
    for _ in range(num_sequences):
        trials = []
        prev_values = []
        
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
            current_value = {"location": location, "identity": identity, "category": category}[task_feature]
            prev_values.append(current_value)
            
            # Three-in-a-row: match if current value equals previous 2 values
            # First 2 trials: no_action (0), later trials: match (2) or no_action (0)
            if len(prev_values) < 3:
                target = 0  # no_action for first 2 trials
            elif prev_values[-1] == prev_values[-2] == prev_values[-3]:
                target = 2  # match
            else:
                target = 0  # no_action (not three-in-a-row)
            
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
    else:
        raise ValueError(f"Unknown task type: {task_config['task_type']}")
