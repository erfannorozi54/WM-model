"""
PyTorch Dataset and DataLoader for N-back working memory experiments.
Wraps the N-back generator in PyTorch-compatible classes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from pathlib import Path
import json

from .nback_generator import NBackGenerator, TaskFeature, Sequence, Trial


class NBackDataset(Dataset):
    """
    PyTorch Dataset for N-back working memory experiments.
    
    Features:
    - On-the-fly sequence generation
    - Image loading and preprocessing
    - Flexible batch configurations
    - Support for mixed N-back conditions
    """
    
    def __init__(self,
                 generator: NBackGenerator,
                 n_values: List[int],
                 task_features: List[TaskFeature],
                 num_sequences: int,
                 sequence_length: int = 6,
                 image_transform: Optional[Callable] = None,
                 cache_sequences: bool = False):
        """
        Initialize the N-back dataset.
        
        Args:
            generator: N-back sequence generator
            n_values: List of N values to use (e.g., [1, 2, 3])
            task_features: List of task features to use
            num_sequences: Total number of sequences in the dataset
            sequence_length: Length of each sequence
            image_transform: Optional image preprocessing transforms
            cache_sequences: Whether to cache generated sequences
        """
        self.generator = generator
        self.n_values = n_values
        self.task_features = task_features
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.cache_sequences = cache_sequences
        
        # Default image transforms
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = image_transform
            
        # Cache for sequences if enabled
        self._sequence_cache = {} if cache_sequences else None
        
        # Pre-generate sequences if caching
        if cache_sequences:
            self._populate_cache()
    
    def _populate_cache(self):
        """Pre-generate and cache all sequences."""
        print(f"Pre-generating {self.num_sequences} sequences...")
        
        sequences = self.generator.generate_mixed_batch(
            batch_size=self.num_sequences,
            n_values=self.n_values,
            task_features=self.task_features,
            sequence_length=self.sequence_length
        )
        
        for i, sequence in enumerate(sequences):
            self._sequence_cache[i] = sequence
            
        print("Sequence caching completed.")
    
    def _get_sequence(self, idx: int) -> Sequence:
        """Get sequence by index, either from cache or generate on-the-fly."""
        if self.cache_sequences and idx in self._sequence_cache:
            return self._sequence_cache[idx]
        else:
            # Generate on-the-fly
            import random
            n = random.choice(self.n_values)
            task_feature = random.choice(self.task_features)
            
            sequence = self.generator.generate_sequence(
                n=n,
                task_feature=task_feature,
                sequence_length=self.sequence_length
            )
            
            if self.cache_sequences:
                self._sequence_cache[idx] = sequence
                
            return sequence
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            # Return black image as fallback
            return torch.zeros(3, 224, 224)
    
    def _encode_response(self, response: str) -> torch.Tensor:
        """
        Encode target response as tensor.
        
        Args:
            response: Response string ("match", "non_match", "no_action")
            
        Returns:
            One-hot encoded response tensor
        """
        response_mapping = {
            "no_action": 0,
            "non_match": 1, 
            "match": 2
        }
        
        encoded = torch.zeros(3)
        encoded[response_mapping.get(response, 0)] = 1.0
        return encoded
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sequence data:
            - images: Tensor of shape (seq_len, 3, H, W)
            - responses: Tensor of shape (seq_len, 3) 
            - task_vector: Tensor of shape (3,)
            - n: Scalar tensor with N value
            - locations: Tensor of shape (seq_len,) with location indices
            - categories: List of category names
            - identities: List of identity names
        """
        sequence = self._get_sequence(idx)
        
        # Load images
        images = []
        responses = []
        locations = []
        categories = []
        identities = []
        
        for trial in sequence.trials:
            # Load and preprocess image
            image = self._load_image(trial.stimulus_path)
            images.append(image)
            
            # Encode response
            response = self._encode_response(trial.target_response)
            responses.append(response)
            
            # Collect metadata
            locations.append(trial.location)
            categories.append(trial.category)
            identities.append(trial.identity)
        
        return {
            'images': torch.stack(images),  # Shape: (seq_len, 3, H, W)
            'responses': torch.stack(responses),  # Shape: (seq_len, 3)
            'task_vector': sequence.task_vector,  # Shape: (3,)
            'n': torch.tensor(sequence.n, dtype=torch.long),  # Shape: ()
            'locations': torch.tensor(locations, dtype=torch.long),  # Shape: (seq_len,)
            'categories': categories,  # List of strings
            'identities': identities,  # List of strings
            'sequence_length': torch.tensor(len(sequence.trials), dtype=torch.long)
        }


class NBackDataModule:
    """
    Data module for managing train/validation/test splits and DataLoaders.
    """
    
    def __init__(self,
                 stimulus_data: Dict[str, Dict],
                 n_values: List[int] = [1, 2, 3],
                 task_features: Optional[List[TaskFeature]] = None,
                 sequence_length: int = 6,
                 batch_size: int = 32,
                 num_train: int = 1000,
                 num_val: int = 200,
                 num_test: int = 200,
                 num_workers: int = 4,
                 image_transform: Optional[Callable] = None):
        """
        Initialize the data module.
        
        Args:
            stimulus_data: Dictionary with stimulus information
            n_values: List of N values for N-back
            task_features: List of task features to use
            sequence_length: Length of each sequence
            batch_size: Batch size for DataLoader
            num_train/val/test: Number of sequences for each split
            num_workers: Number of DataLoader workers
            image_transform: Optional image preprocessing
        """
        self.stimulus_data = stimulus_data
        self.n_values = n_values
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if task_features is None:
            self.task_features = [TaskFeature.LOCATION, TaskFeature.IDENTITY, TaskFeature.CATEGORY]
        else:
            self.task_features = task_features
        
        # Create generator
        self.generator = NBackGenerator(
            stimulus_data=stimulus_data,
            sequence_length=sequence_length
        )
        
        # Create datasets
        self.train_dataset = NBackDataset(
            generator=self.generator,
            n_values=n_values,
            task_features=self.task_features,
            num_sequences=num_train,
            sequence_length=sequence_length,
            image_transform=image_transform,
            cache_sequences=False  # Don't cache training data for variety
        )
        
        self.val_dataset = NBackDataset(
            generator=self.generator,
            n_values=n_values,
            task_features=self.task_features,
            num_sequences=num_val,
            sequence_length=sequence_length,
            image_transform=image_transform,
            cache_sequences=True  # Cache validation data for consistency
        )
        
        self.test_dataset = NBackDataset(
            generator=self.generator,
            n_values=n_values,
            task_features=self.task_features,
            num_sequences=num_test,
            sequence_length=sequence_length,
            image_transform=image_transform,
            cache_sequences=True  # Cache test data for consistency
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def sample_batch(self, split: str = "train") -> Dict[str, torch.Tensor]:
        """
        Get a sample batch from specified split.
        
        Args:
            split: Which split to sample from ("train", "val", "test")
            
        Returns:
            Sample batch dictionary
        """
        if split == "train":
            dataloader = self.train_dataloader()
        elif split == "val":
            dataloader = self.val_dataloader()
        elif split == "test":
            dataloader = self.test_dataloader()
        else:
            raise ValueError(f"Unknown split: {split}")
        
        return next(iter(dataloader))


def create_demo_dataset() -> NBackDataModule:
    """
    Create a demo dataset for testing.
    
    Returns:
        Demo data module
    """
    from .shapenet_downloader import create_sample_stimulus_data
    
    # Create sample stimulus data
    stimulus_data = create_sample_stimulus_data()
    
    # Create data module
    data_module = NBackDataModule(
        stimulus_data=stimulus_data,
        n_values=[1, 2],
        task_features=[TaskFeature.LOCATION, TaskFeature.CATEGORY],
        sequence_length=6,
        batch_size=4,
        num_train=20,
        num_val=10,
        num_test=10,
        num_workers=0  # Disable multiprocessing for demo
    )
    
    return data_module


if __name__ == "__main__":
    print("N-back Dataset Demo")
    print("=" * 25)
    
    # Create demo dataset
    data_module = create_demo_dataset()
    
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Val dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    
    # Sample a batch
    print("\nSampling training batch...")
    try:
        batch = data_module.sample_batch("train")
        
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Images shape: {batch['images'].shape}")
        print(f"Responses shape: {batch['responses'].shape}")
        print(f"Task vectors shape: {batch['task_vector'].shape}")
        print(f"N values: {batch['n']}")
        
        print("\nFirst sequence in batch:")
        print(f"  N-back: {batch['n'][0].item()}")
        print(f"  Task vector: {batch['task_vector'][0]}")
        print(f"  Categories: {batch['categories'][0]}")
        print(f"  Locations: {batch['locations'][0]}")
        
    except Exception as e:
        print(f"Error sampling batch: {e}")
        print("This is expected if stimulus images don't exist yet.")
    
    print("\nDataset demo completed!")
