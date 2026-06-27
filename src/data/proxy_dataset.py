"""
PyTorch Dataset and DataLoader for proxy task pre-training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
from typing import List, Dict, Optional, Callable

from .proxy_generator import ProxyTaskGenerator, ProxySequence, FEATURE_NAMES
from .nback_generator import TaskFeature

from ..utils.logger import get_logger
logger = get_logger()


class ProxyNBackDataset(Dataset):
    def __init__(self, generator: ProxyTaskGenerator,
                 n_values: List[int],
                 task_features: List[str],
                 num_sequences: int,
                 sequence_length: int = 6,
                 image_transform: Optional[Callable] = None,
                 cache_sequences: bool = False,
                 match_probability: float = 0.5,
                 balanced: bool = True):
        self.generator = generator
        self.n_values = n_values
        self.task_features = task_features
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.cache_sequences = cache_sequences
        self.match_probability = match_probability
        self.balanced = balanced

        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = image_transform

        self._sequence_cache = {} if cache_sequences else None
        if cache_sequences:
            self._populate_cache()

    def _populate_cache(self):
        logger.info(f"Pre-generating {self.num_sequences} proxy sequences...")
        if self.balanced:
            per_task = max(1, self.num_sequences // (len(self.task_features) * len(self.n_values)))
            sequences = self.generator.generate_all_task_vectors_batch(
                batch_size_per_task=per_task,
                n_values=self.n_values,
                task_features=self.task_features,
                sequence_length=self.sequence_length,
                match_probability=self.match_probability,
            )
            while len(sequences) < self.num_sequences:
                n = random.choice(self.n_values)
                tf = random.choice(self.task_features)
                sequences.append(self.generator.generate_sequence(
                    n, tf, self.sequence_length, self.match_probability))
            sequences = sequences[:self.num_sequences]
        else:
            sequences = self.generator.generate_mixed_batch(
                batch_size=self.num_sequences,
                n_values=self.n_values,
                task_features=self.task_features,
                sequence_length=self.sequence_length,
                match_probability=self.match_probability,
            )
        for i, seq in enumerate(sequences):
            self._sequence_cache[i] = seq
        logger.info("Proxy sequence caching completed.")

    def _get_sequence(self, idx: int) -> ProxySequence:
        if self.cache_sequences and idx in self._sequence_cache:
            return self._sequence_cache[idx]
        n = random.choice(self.n_values)
        tf = random.choice(self.task_features)
        seq = self.generator.generate_sequence(n, tf, self.sequence_length, self.match_probability)
        if self.cache_sequences:
            self._sequence_cache[idx] = seq
        return seq

    def _load_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            logger.warning(f"Could not load image {image_path}: {e}")
            return torch.zeros(3, 224, 224)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self._get_sequence(idx)

        images = []
        locations = []
        categories = []
        identities = []

        for trial in seq.trials:
            image = self._load_image(trial['stimulus_path'])
            images.append(image)
            locations.append(trial['location'])
            categories.append(trial['category'])
            identities.append(trial['identity'])

        proxy_targets = torch.tensor(seq.proxy_targets, dtype=torch.long)

        if seq.proxy_targets_1back is not None:
            targets_1back = torch.tensor(seq.proxy_targets_1back, dtype=torch.long)
            targets_3back = torch.tensor(seq.proxy_targets_3back, dtype=torch.long)
        else:
            targets_1back = torch.full_like(proxy_targets, -1)
            targets_3back = torch.full_like(proxy_targets, -1)

        return {
            'images': torch.stack(images),
            'proxy_targets': proxy_targets,
            'proxy_targets_1back': targets_1back,
            'proxy_targets_3back': targets_3back,
            'task_vector': seq.task_vector,
            'n': torch.tensor(seq.n, dtype=torch.long),
            'locations': torch.tensor(locations, dtype=torch.long),
            'categories': categories,
            'identities': identities,
            'task_feature': seq.task_feature,
            'num_classes': seq.num_classes,
            'sequence_length': torch.tensor(len(seq.trials), dtype=torch.long),
        }


def proxy_collate_fn(batch):
    images = torch.stack([item['images'] for item in batch])
    proxy_targets = torch.stack([item['proxy_targets'] for item in batch])
    task_vector = torch.stack([item['task_vector'] for item in batch])
    n = torch.stack([item['n'] for item in batch])
    locations = torch.stack([item['locations'] for item in batch])
    sequence_length = torch.stack([item['sequence_length'] for item in batch])
    num_classes = [item['num_classes'] for item in batch]
    task_features = [item['task_feature'] for item in batch]
    categories = [item['categories'] for item in batch]
    identities = [item['identities'] for item in batch]

    return {
        'images': images,
        'proxy_targets': proxy_targets,
        'proxy_targets_1back': torch.stack([item['proxy_targets_1back'] for item in batch]),
        'proxy_targets_3back': torch.stack([item['proxy_targets_3back'] for item in batch]),
        'task_vector': task_vector,
        'n': n,
        'locations': locations,
        'categories': categories,
        'identities': identities,
        'task_feature': task_features,
        'num_classes': num_classes,
        'sequence_length': sequence_length,
    }


class ProxyDataModule:
    def __init__(self, stimulus_data: Dict,
                 val_novel_angle_data: Optional[Dict] = None,
                 val_novel_identity_data: Optional[Dict] = None,
                 n_values: List[int] = [1, 2, 3],
                 task_features: Optional[List[str]] = None,
                 sequence_length: int = 6,
                 batch_size: int = 32,
                 num_train: int = 30000,
                 num_val_novel_angle: int = 400,
                 num_val_novel_identity: int = 400,
                 num_workers: int = 4,
                 match_probability: float = 0.5,
                 cache_train_sequences: bool = False,
                 cache_val_sequences: bool = True,
                 identity_mapping: Optional[Dict[str, int]] = None):
        self.stimulus_data = stimulus_data
        self.n_values = n_values
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        if task_features is None:
            self.task_features = ["location", "identity", "category"]
        else:
            self.task_features = task_features

        self.train_generator = ProxyTaskGenerator(
            stimulus_data=stimulus_data,
            sequence_length=sequence_length,
            identity_mapping=identity_mapping,
        )

        self.train_dataset = ProxyNBackDataset(
            generator=self.train_generator,
            n_values=n_values,
            task_features=self.task_features,
            num_sequences=num_train,
            sequence_length=sequence_length,
            cache_sequences=cache_train_sequences,
            match_probability=match_probability,
            balanced=True,
        )

        if val_novel_angle_data:
            self.val_novel_angle_generator = ProxyTaskGenerator(
                stimulus_data=val_novel_angle_data,
                sequence_length=sequence_length,
                identity_mapping=self.train_generator.identity_mapping,
            )
            self.val_novel_angle_dataset = ProxyNBackDataset(
                generator=self.val_novel_angle_generator,
                n_values=n_values,
                task_features=self.task_features,
                num_sequences=num_val_novel_angle,
                sequence_length=sequence_length,
                cache_sequences=cache_val_sequences,
                match_probability=match_probability,
                balanced=True,
            )
        else:
            self.val_novel_angle_dataset = None

        if val_novel_identity_data:
            self.val_novel_identity_generator = ProxyTaskGenerator(
                stimulus_data=val_novel_identity_data,
                sequence_length=sequence_length,
                identity_mapping=self.train_generator.identity_mapping,
            )
            self.val_novel_identity_dataset = ProxyNBackDataset(
                generator=self.val_novel_identity_generator,
                n_values=n_values,
                task_features=self.task_features,
                num_sequences=num_val_novel_identity,
                sequence_length=sequence_length,
                cache_sequences=cache_val_sequences,
                match_probability=match_probability,
                balanced=True,
            )
        else:
            self.val_novel_identity_dataset = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=proxy_collate_fn,
        )

    def val_novel_angle_dataloader(self) -> Optional[DataLoader]:
        if self.val_novel_angle_dataset is None:
            return None
        return DataLoader(
            self.val_novel_angle_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=proxy_collate_fn,
        )

    def val_novel_identity_dataloader(self) -> Optional[DataLoader]:
        if self.val_novel_identity_dataset is None:
            return None
        return DataLoader(
            self.val_novel_identity_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=proxy_collate_fn,
        )
