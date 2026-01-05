"""
N-back trial generator for working memory experiments.
Generates sequences with match/non-match conditions for Location, Identity, or Category tasks.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import random
from dataclasses import dataclass
from enum import Enum


class TaskFeature(Enum):
    """Task feature types for N-back experiments."""
    LOCATION = "location"
    IDENTITY = "identity" 
    CATEGORY = "category"


@dataclass
class Trial:
    """Single trial in an N-back sequence."""
    stimulus_path: str
    location: int
    category: str
    identity: str
    target_response: str  # "match", "non_match", or "no_action"
    n: int
    trial_index: int


@dataclass
class Sequence:
    """Complete N-back sequence."""
    trials: List[Trial]
    task_feature: TaskFeature
    n: int
    sequence_length: int
    task_vector: torch.Tensor


class NBackGenerator:
    """
    Generates N-back task sequences for working memory experiments.
    
    Features:
    - Configurable N (1-back, 2-back, 3-back)
    - Three task types: Location, Identity, Category
    - Flexible sequence length
    - Batch generation
    - Task identity vectors
    """
    
    def __init__(self,
                 stimulus_data: Dict[str, Dict],
                 n_locations: int = 4,
                 sequence_length: int = 6,
                 match_probability: float = 0.3):
        """
        Initialize the N-back generator.
        
        Args:
            stimulus_data: Dictionary with stimulus information
                         Format: {category: {identity: [stimulus_paths]}}
            n_locations: Number of possible stimulus locations
            sequence_length: Length of each trial sequence
            match_probability: Probability of match trials
        """
        self.stimulus_data = stimulus_data
        self.n_locations = n_locations
        self.sequence_length = sequence_length
        self.match_probability = match_probability
        
        # Extract categories and identities
        self.categories = list(stimulus_data.keys())
        self.identities = {}
        self.all_stimuli = []
        
        for category, identities in stimulus_data.items():
            self.identities[category] = list(identities.keys())
            for identity, stimuli in identities.items():
                for stimulus_path in stimuli:
                    self.all_stimuli.append({
                        'path': stimulus_path,
                        'category': category,
                        'identity': identity,
                        'location': self._extract_location_from_path(stimulus_path)
                    })
    
    def _extract_location_from_path(self, stimulus_path: str) -> int:
        """Extract location index from stimulus filename."""
        # Assumes filename contains "_loc{N}_" pattern
        try:
            path_str = str(stimulus_path)
            loc_start = path_str.find("_loc") + 4
            loc_end = path_str.find("_", loc_start)
            return int(path_str[loc_start:loc_end])
        except:
            return random.randint(0, self.n_locations - 1)
    
    def _create_task_vector(self, task_feature: TaskFeature, n: int) -> torch.Tensor:
        """
        Create task identity vector encoding both feature and N.
        
        Format: 6-digit vector [feature(3), n(3)]
        - First 3: one-hot feature (location/identity/category)
        - Last 3: one-hot N value (1-back/2-back/3-back)
        
        Args:
            task_feature: Type of task feature
            n: N-back value (1, 2, or 3)
            
        Returns:
            Task vector of shape (6,)
        """
        task_mapping = {
            TaskFeature.LOCATION: 0,
            TaskFeature.IDENTITY: 1, 
            TaskFeature.CATEGORY: 2
        }
        
        vector = torch.zeros(6)
        # Feature encoding (first 3)
        vector[task_mapping[task_feature]] = 1.0
        # N encoding (last 3): n=1 -> index 3, n=2 -> index 4, n=3 -> index 5
        if 1 <= n <= 3:
            vector[2 + n] = 1.0
        return vector
    
    def _get_match_stimulus(self, 
                           reference_trial: Trial,
                           task_feature: TaskFeature) -> Dict:
        """
        Find a stimulus that matches the reference trial on the specified feature.
        
        Args:
            reference_trial: Trial to match against
            task_feature: Feature to match on
            
        Returns:
            Dictionary with stimulus information
        """
        matching_stimuli = []
        
        for stimulus in self.all_stimuli:
            if task_feature == TaskFeature.LOCATION:
                if stimulus['location'] == reference_trial.location:
                    matching_stimuli.append(stimulus)
            elif task_feature == TaskFeature.IDENTITY:
                if (stimulus['identity'] == reference_trial.identity and 
                    stimulus['category'] == reference_trial.category):
                    matching_stimuli.append(stimulus)
            elif task_feature == TaskFeature.CATEGORY:
                if stimulus['category'] == reference_trial.category:
                    matching_stimuli.append(stimulus)
        
        if matching_stimuli:
            return random.choice(matching_stimuli)
        else:
            # Fallback to random stimulus
            return random.choice(self.all_stimuli)
    
    def _get_non_match_stimulus(self,
                               reference_trial: Trial,
                               task_feature: TaskFeature) -> Dict:
        """
        Find a stimulus that does NOT match the reference trial on the specified feature.
        
        Args:
            reference_trial: Trial to avoid matching
            task_feature: Feature to avoid matching on
            
        Returns:
            Dictionary with stimulus information
        """
        non_matching_stimuli = []
        
        for stimulus in self.all_stimuli:
            if task_feature == TaskFeature.LOCATION:
                if stimulus['location'] != reference_trial.location:
                    non_matching_stimuli.append(stimulus)
            elif task_feature == TaskFeature.IDENTITY:
                if not (stimulus['identity'] == reference_trial.identity and 
                       stimulus['category'] == reference_trial.category):
                    non_matching_stimuli.append(stimulus)
            elif task_feature == TaskFeature.CATEGORY:
                if stimulus['category'] != reference_trial.category:
                    non_matching_stimuli.append(stimulus)
        
        if non_matching_stimuli:
            return random.choice(non_matching_stimuli)
        else:
            # Fallback to random stimulus
            return random.choice(self.all_stimuli)
    
    def generate_sequence(self,
                         n: int,
                         task_feature: TaskFeature,
                         sequence_length: Optional[int] = None) -> Sequence:
        """
        Generate a single N-back sequence.
        
        Args:
            n: N for N-back (1, 2, 3, etc.)
            task_feature: Feature to use for matching
            sequence_length: Override default sequence length
            
        Returns:
            Generated sequence
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        trials = []
        
        # Generate initial trials (no matches possible)
        for i in range(min(n, sequence_length)):
            stimulus = random.choice(self.all_stimuli)
            trial = Trial(
                stimulus_path=stimulus['path'],
                location=stimulus['location'],
                category=stimulus['category'],
                identity=stimulus['identity'],
                target_response="no_action",
                n=n,
                trial_index=i
            )
            trials.append(trial)
        
        # Generate remaining trials with possible matches
        for i in range(n, sequence_length):
            reference_trial = trials[i - n]
            
            # Decide if this should be a match trial
            is_match = random.random() < self.match_probability
            
            if is_match:
                stimulus = self._get_match_stimulus(reference_trial, task_feature)
                target_response = "match"
            else:
                stimulus = self._get_non_match_stimulus(reference_trial, task_feature)
                target_response = "non_match"
            
            trial = Trial(
                stimulus_path=stimulus['path'],
                location=stimulus['location'],
                category=stimulus['category'],
                identity=stimulus['identity'],
                target_response=target_response,
                n=n,
                trial_index=i
            )
            trials.append(trial)
        
        # Create task vector (includes both feature and N)
        task_vector = self._create_task_vector(task_feature, n)
        
        return Sequence(
            trials=trials,
            task_feature=task_feature,
            n=n,
            sequence_length=sequence_length,
            task_vector=task_vector
        )
    
    def generate_batch(self,
                      batch_size: int,
                      n: int,
                      task_feature: TaskFeature,
                      sequence_length: Optional[int] = None) -> List[Sequence]:
        """
        Generate a batch of N-back sequences.
        
        Args:
            batch_size: Number of sequences to generate
            n: N for N-back
            task_feature: Feature to use for matching
            sequence_length: Override default sequence length
            
        Returns:
            List of generated sequences
        """
        sequences = []
        
        for _ in range(batch_size):
            sequence = self.generate_sequence(n, task_feature, sequence_length)
            sequences.append(sequence)
            
        return sequences
    
    def generate_mixed_batch(self,
                            batch_size: int,
                            n_values: List[int],
                            task_features: List[TaskFeature],
                            sequence_length: Optional[int] = None) -> List[Sequence]:
        """
        Generate a batch with mixed N values and task features.
        
        Args:
            batch_size: Total number of sequences
            n_values: List of N values to sample from
            task_features: List of task features to sample from
            sequence_length: Override default sequence length
            
        Returns:
            List of generated sequences
        """
        sequences = []
        
        for _ in range(batch_size):
            n = random.choice(n_values)
            task_feature = random.choice(task_features)
            sequence = self.generate_sequence(n, task_feature, sequence_length)
            sequences.append(sequence)
            
        return sequences
    
    def get_statistics(self, sequences: List[Sequence]) -> Dict:
        """
        Get statistics about generated sequences.
        
        Args:
            sequences: List of sequences to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_sequences': len(sequences),
            'n_distribution': {},
            'task_distribution': {},
            'response_distribution': {},
            'match_rate': 0.0
        }
        
        total_trials = 0
        total_matches = 0
        
        for sequence in sequences:
            # N distribution
            n_str = str(sequence.n)
            stats['n_distribution'][n_str] = stats['n_distribution'].get(n_str, 0) + 1
            
            # Task distribution
            task_str = sequence.task_feature.value
            stats['task_distribution'][task_str] = stats['task_distribution'].get(task_str, 0) + 1
            
            # Response distribution
            for trial in sequence.trials:
                total_trials += 1
                response = trial.target_response
                stats['response_distribution'][response] = stats['response_distribution'].get(response, 0) + 1
                
                if response == "match":
                    total_matches += 1
        
        if total_trials > 0:
            stats['match_rate'] = total_matches / total_trials
            
        return stats


if __name__ == "__main__":
    # Demo usage
    print("N-back Generator Demo")
    print("=" * 25)
    
    # Create sample stimulus data
    from .shapenet_downloader import create_sample_stimulus_data
    stimulus_data = create_sample_stimulus_data()
    
    # Initialize generator
    generator = NBackGenerator(stimulus_data)
    
    # Generate a sample sequence
    sequence = generator.generate_sequence(
        n=2,
        task_feature=TaskFeature.LOCATION,
        sequence_length=6
    )
    
    print(f"Generated {sequence.n}-back sequence for {sequence.task_feature.value}")
    print(f"Sequence length: {len(sequence.trials)}")
    print(f"Task vector: {sequence.task_vector}")
    
    print("\nTrials:")
    for i, trial in enumerate(sequence.trials):
        print(f"  {i}: {trial.target_response} - {Path(trial.stimulus_path).name}")
    
    # Generate batch
    batch = generator.generate_batch(
        batch_size=3,
        n=1,
        task_feature=TaskFeature.CATEGORY
    )
    
    print(f"\nGenerated batch of {len(batch)} sequences")
    
    # Get statistics
    stats = generator.get_statistics(batch)
    print(f"Statistics: {stats}")
