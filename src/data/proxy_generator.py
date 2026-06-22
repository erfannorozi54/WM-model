"""
Proxy Task Generator for Working Memory Pre-training.

Instead of the standard N-back match/non-match/no_action classification,
the proxy task asks the model to recall the feature value from N steps back:
- Location: predict which of 4 locations the stimulus N steps ago was at
- Identity: predict which identity the stimulus N steps ago had
- Category: predict which category the stimulus N steps ago belonged to

This provides a richer training signal that forces the model to actually
encode specific feature values in working memory.
"""

import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .nback_generator import TaskFeature, NBackGenerator

from ..utils.logger import get_logger
logger = get_logger()

FEATURE_NAMES = ["location", "identity", "category"]
FEATURE_IDX = {"location": 0, "identity": 1, "category": 2}


def build_identity_mapping(stimulus_data: Dict) -> Tuple[Dict[str, int], int]:
    mapping = {}
    idx = 0
    for category in sorted(stimulus_data.keys()):
        for identity in sorted(stimulus_data[category].keys()):
            key = f"{category}_{identity}" if not identity.startswith(category) else identity
            if key not in mapping:
                mapping[key] = idx
                idx += 1
    return mapping, idx


def _identity_key(category: str, identity: str) -> str:
    if identity.startswith(category):
        return identity
    return f"{category}_{identity}"


class ProxySequence:
    def __init__(self, trials, task_feature: str, n: int, sequence_length: int,
                 task_vector: torch.Tensor, proxy_targets: List[int],
                 num_classes: int):
        self.trials = trials
        self.task_feature = task_feature
        self.n = n
        self.sequence_length = sequence_length
        self.task_vector = task_vector
        self.proxy_targets = proxy_targets
        self.num_classes = num_classes


class ProxyTaskGenerator:
    def __init__(self, stimulus_data: Dict, n_locations: int = 4,
                 sequence_length: int = 6, identity_mapping: Optional[Dict[str, int]] = None):
        self.stimulus_data = stimulus_data
        self.n_locations = n_locations
        self.sequence_length = sequence_length

        self.categories = sorted(stimulus_data.keys())
        self.num_categories = len(self.categories)
        self.category_to_idx = {c: i for i, c in enumerate(self.categories)}

        self.all_stimuli = []
        self.identities_per_category = {}
        for category in self.categories:
            identities = sorted(stimulus_data[category].keys())
            self.identities_per_category[category] = identities
            for identity in identities:
                for stimulus_path in stimulus_data[category][identity]:
                    self.all_stimuli.append({
                        'path': stimulus_path,
                        'category': category,
                        'identity': identity,
                        'location': self._extract_location(stimulus_path),
                    })

        if identity_mapping is None:
            identity_mapping, _ = build_identity_mapping(stimulus_data)
        self.identity_mapping = identity_mapping
        self.num_identities = len(identity_mapping)

        self.num_classes = {
            "location": n_locations,
            "identity": self.num_identities,
            "category": self.num_categories,
        }

    def _extract_location(self, path: str) -> int:
        try:
            path_str = str(path)
            loc_start = path_str.find("_loc") + 4
            loc_end = path_str.find("_", loc_start)
            return int(path_str[loc_start:loc_end])
        except Exception:
            return random.randint(0, self.n_locations - 1)

    def _create_task_vector(self, task_feature: str, n: int) -> torch.Tensor:
        vector = torch.zeros(6)
        feat_idx = FEATURE_IDX.get(task_feature, 0)
        vector[feat_idx] = 1.0
        if 1 <= n <= 3:
            vector[2 + n] = 1.0
        return vector

    def _get_feature_value(self, stimulus: Dict, task_feature: str) -> int:
        if task_feature == "location":
            return stimulus['location']
        elif task_feature == "identity":
            key = _identity_key(stimulus['category'], stimulus['identity'])
            return self.identity_mapping.get(key, 0)
        elif task_feature == "category":
            return self.category_to_idx.get(stimulus['category'], 0)
        return 0

    def _pick_stimulus_with_constraint(self, task_feature: str, n_back_stim: Dict,
                                        force_match: bool) -> Dict:
        if task_feature == "location":
            ref_val = n_back_stim['location']
            if force_match:
                candidates = [s for s in self.all_stimuli if s['location'] == ref_val]
            else:
                candidates = [s for s in self.all_stimuli if s['location'] != ref_val]
        elif task_feature == "identity":
            ref_key = _identity_key(n_back_stim['category'], n_back_stim['identity'])
            if force_match:
                candidates = [s for s in self.all_stimuli
                              if _identity_key(s['category'], s['identity']) == ref_key]
            else:
                candidates = [s for s in self.all_stimuli
                              if _identity_key(s['category'], s['identity']) != ref_key]
        elif task_feature == "category":
            ref_cat = n_back_stim['category']
            if force_match:
                candidates = [s for s in self.all_stimuli if s['category'] == ref_cat]
            else:
                candidates = [s for s in self.all_stimuli if s['category'] != ref_cat]
        else:
            candidates = self.all_stimuli

        if candidates:
            return random.choice(candidates)
        return random.choice(self.all_stimuli)

    def generate_sequence(self, n: int, task_feature: str,
                          sequence_length: Optional[int] = None,
                          match_probability: float = 0.5) -> ProxySequence:
        if sequence_length is None:
            sequence_length = self.sequence_length

        trials = []
        proxy_targets = []

        for i in range(min(n, sequence_length)):
            stimulus = random.choice(self.all_stimuli)
            trials.append({
                'stimulus_path': stimulus['path'],
                'location': stimulus['location'],
                'category': stimulus['category'],
                'identity': stimulus['identity'],
                'trial_index': i,
            })
            proxy_targets.append(-1)

        for i in range(n, sequence_length):
            n_back_stim = trials[i - n]
            is_match = random.random() < match_probability
            stimulus = self._pick_stimulus_with_constraint(task_feature, n_back_stim, is_match)

            trials.append({
                'stimulus_path': stimulus['path'],
                'location': stimulus['location'],
                'category': stimulus['category'],
                'identity': stimulus['identity'],
                'trial_index': i,
            })

            target_value = self._get_feature_value(n_back_stim, task_feature)
            proxy_targets.append(target_value)

        task_vector = self._create_task_vector(task_feature, n)
        num_classes = self.num_classes[task_feature]

        return ProxySequence(
            trials=trials,
            task_feature=task_feature,
            n=n,
            sequence_length=sequence_length,
            task_vector=task_vector,
            proxy_targets=proxy_targets,
            num_classes=num_classes,
        )

    def generate_novel_proxy_sequence(self, task_name: str,
                                       sequence_length: Optional[int] = None) -> ProxySequence:
        if sequence_length is None:
            sequence_length = self.sequence_length

        if task_name == "nback_4":
            return self._generate_standard_proxy(4, "location", sequence_length, novel=True)
        elif task_name == "nback_5":
            return self._generate_standard_proxy(5, "location", sequence_length, novel=True)
        elif task_name == "three_in_a_row":
            return self._generate_three_in_a_row_proxy(sequence_length)
        elif task_name == "alternating":
            return self._generate_alternating_proxy(sequence_length)
        else:
            raise ValueError(f"Unknown novel task: {task_name}")

    def _generate_standard_proxy(self, n: int, task_feature: str,
                                  sequence_length: int, novel: bool = False) -> ProxySequence:
        seq = self.generate_sequence(n, task_feature, sequence_length)
        if novel:
            tv = torch.zeros(6)
            tv[FEATURE_IDX[task_feature]] = 1.0
            if n == 4:
                tv[3] = 1.0
                tv[4] = 1.0
            elif n == 5:
                tv[3] = 1.0
                tv[5] = 1.0
            seq.task_vector = tv
        return seq

    def _generate_three_in_a_row_proxy(self, sequence_length: int) -> ProxySequence:
        trials = []
        proxy_targets = []
        history = []

        for t in range(sequence_length):
            stimulus = random.choice(self.all_stimuli)
            trials.append({
                'stimulus_path': stimulus['path'],
                'location': stimulus['location'],
                'category': stimulus['category'],
                'identity': stimulus['identity'],
                'trial_index': t,
            })
            history.append(stimulus['location'])

            if t < 1:
                proxy_targets.append(-1)
            else:
                target = 1 if history[t] == history[t - 1] else 0
                proxy_targets.append(target)

        task_vector = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        return ProxySequence(
            trials=trials, task_feature="location", n=0,
            sequence_length=sequence_length, task_vector=task_vector,
            proxy_targets=proxy_targets, num_classes=2,
        )

    def _generate_alternating_proxy(self, sequence_length: int, n: int = 2) -> ProxySequence:
        trials = []
        proxy_targets = []
        location_history = []
        identity_history = []

        for t in range(sequence_length):
            stimulus = random.choice(self.all_stimuli)
            trials.append({
                'stimulus_path': stimulus['path'],
                'location': stimulus['location'],
                'category': stimulus['category'],
                'identity': stimulus['identity'],
                'trial_index': t,
            })
            location_history.append(stimulus['location'])
            identity_history.append(_identity_key(stimulus['category'], stimulus['identity']))

            if t < n:
                proxy_targets.append(-1)
            elif t % 2 == 0:
                ref_loc = location_history[t - n]
                target = 1 if stimulus['location'] == ref_loc else 0
                proxy_targets.append(target)
            else:
                ref_id = identity_history[t - n]
                cur_id = _identity_key(stimulus['category'], stimulus['identity'])
                target = 1 if cur_id == ref_id else 0
                proxy_targets.append(target)

        task_vector = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        return ProxySequence(
            trials=trials, task_feature="alternating", n=n,
            sequence_length=sequence_length, task_vector=task_vector,
            proxy_targets=proxy_targets, num_classes=2,
        )

    def generate_mixed_batch(self, batch_size: int,
                             n_values: List[int],
                             task_features: List[str],
                             sequence_length: Optional[int] = None,
                             match_probability: float = 0.5) -> List[ProxySequence]:
        sequences = []
        for _ in range(batch_size):
            n = random.choice(n_values)
            task_feature = random.choice(task_features)
            seq = self.generate_sequence(n, task_feature, sequence_length, match_probability)
            sequences.append(seq)
        return sequences

    def generate_all_task_vectors_batch(self, batch_size_per_task: int,
                                         n_values: List[int] = [1, 2, 3],
                                         task_features: List[str] = None,
                                         sequence_length: Optional[int] = None,
                                         match_probability: float = 0.5) -> List[ProxySequence]:
        if task_features is None:
            task_features = ["location", "identity", "category"]

        sequences = []
        for tf in task_features:
            for n in n_values:
                for _ in range(batch_size_per_task):
                    seq = self.generate_sequence(n, tf, sequence_length, match_probability)
                    sequences.append(seq)
        return sequences
