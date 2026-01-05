"""
Validation data splits for generalization testing.

Implements two types of validation as described in the paper:
1. Novel Angles: Same identities as training, but new viewing angles
2. Novel Identities: New object identities from same categories
"""

from typing import Dict, List, Tuple, Set
from pathlib import Path
import re


class ValidationSplitter:
    """
    Splits stimulus data into training and two validation sets:
    - Novel Angles: Same identities, different angles
    - Novel Identities: Different identities, same categories
    """
    
    def __init__(self, 
                 all_stimulus_data: Dict[str, Dict[str, List[str]]],
                 train_angles: List[int] = [0, 1, 2],  # angles 0,1,2 for training
                 val_angles: List[int] = [3],           # angle 3 for novel-angle validation
                 train_identity_ratio: float = 0.6):    # 60% identities for training
        """
        Initialize the validation splitter.
        
        Args:
            all_stimulus_data: All available stimuli
                              Format: {category: {identity: [paths]}}
            train_angles: Angle indices to use for training (e.g., [0,1,2])
            val_angles: Angle indices to use for novel-angle validation (e.g., [3])
            train_identity_ratio: Fraction of identities for training (rest for validation)
        """
        self.all_stimulus_data = all_stimulus_data
        self.train_angles = train_angles
        self.val_angles = val_angles
        self.train_identity_ratio = train_identity_ratio
        
    def _parse_stimulus_path(self, path: str) -> Tuple[str, str, int, int]:
        """
        Parse stimulus filename to extract metadata.
        
        Args:
            path: Path like 'data/stimuli/stimulus_airplane_000_loc2_angle3.png'
            
        Returns:
            Tuple of (category, identity, location, angle)
        """
        filename = Path(path).stem  # Get filename without extension
        # Pattern: stimulus_{category}_{id}_{loc}{loc_num}_angle{angle_num}
        match = re.match(r'stimulus_(\w+)_(\d+)_loc(\d+)_angle(\d+)', filename)
        
        if match:
            category = match.group(1)
            identity_num = match.group(2)
            location = int(match.group(3))
            angle = int(match.group(4))
            identity = f"{category}_{identity_num}"
            return category, identity, location, angle
        else:
            raise ValueError(f"Cannot parse stimulus path: {path}")
    
    def create_splits(self) -> Tuple[Dict, Dict, Dict]:
        """
        Create three data splits: training, novel-angle validation, novel-identity validation.
        
        Returns:
            Tuple of (train_data, val_novel_angle_data, val_novel_identity_data)
            Each is a dict in format: {category: {identity: [paths]}}
        """
        train_data = {}
        val_novel_angle_data = {}
        val_novel_identity_data = {}
        
        for category, identities_dict in self.all_stimulus_data.items():
            # Get sorted list of identities
            all_identities = sorted(identities_dict.keys())
            num_identities = len(all_identities)
            
            # Calculate split based on ratio (at least 1 for each split)
            num_train = max(1, int(num_identities * self.train_identity_ratio))
            num_val = num_identities - num_train
            
            if num_val < 1:
                raise ValueError(
                    f"Category {category} has only {num_identities} identities, "
                    f"need at least 2 for train/val split"
                )
            
            # Split identities
            train_identities = all_identities[:num_train]
            val_identities = all_identities[num_train:]
            
            # Initialize category dictionaries
            train_data[category] = {}
            val_novel_angle_data[category] = {}
            val_novel_identity_data[category] = {}
            
            # Process training identities
            for identity in train_identities:
                paths = identities_dict[identity]
                
                # Separate by angle
                train_paths = []
                val_angle_paths = []
                
                for path in paths:
                    _, _, _, angle = self._parse_stimulus_path(path)
                    
                    if angle in self.train_angles:
                        train_paths.append(path)
                    elif angle in self.val_angles:
                        val_angle_paths.append(path)
                
                if train_paths:
                    train_data[category][identity] = train_paths
                if val_angle_paths:
                    val_novel_angle_data[category][identity] = val_angle_paths
            
            # Process validation identities (use all angles)
            for identity in val_identities:
                paths = identities_dict[identity]
                val_novel_identity_data[category][identity] = paths
        
        return train_data, val_novel_angle_data, val_novel_identity_data
    
    def get_split_statistics(self, 
                           train_data: Dict,
                           val_novel_angle_data: Dict, 
                           val_novel_identity_data: Dict) -> Dict:
        """
        Get statistics about the data splits.
        
        Returns:
            Dictionary with split statistics
        """
        def count_stimuli(data):
            total = 0
            identities = 0
            for category, id_dict in data.items():
                identities += len(id_dict)
                for paths in id_dict.values():
                    total += len(paths)
            return total, identities
        
        train_stimuli, train_ids = count_stimuli(train_data)
        val_angle_stimuli, val_angle_ids = count_stimuli(val_novel_angle_data)
        val_id_stimuli, val_id_ids = count_stimuli(val_novel_identity_data)
        
        return {
            'training': {
                'num_stimuli': train_stimuli,
                'num_identities': train_ids,
                'angles': self.train_angles,
                'categories': list(train_data.keys())
            },
            'val_novel_angle': {
                'num_stimuli': val_angle_stimuli,
                'num_identities': val_angle_ids,
                'angles': self.val_angles,
                'categories': list(val_novel_angle_data.keys()),
                'note': 'Same identities as training, but novel viewing angles'
            },
            'val_novel_identity': {
                'num_stimuli': val_id_stimuli,
                'num_identities': val_id_ids,
                'angles': list(range(4)),  # All angles
                'categories': list(val_novel_identity_data.keys()),
                'note': 'Novel object identities from same categories'
            }
        }


def load_and_split_stimuli(
    stimuli_dir: str = "data/stimuli",
    train_angles: List[int] = [0, 1, 2],
    val_angles: List[int] = [3],
    train_identity_ratio: float = 0.6
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load all stimuli and create training/validation splits.
    
    Args:
        stimuli_dir: Directory containing stimulus images
        train_angles: Angle indices for training
        val_angles: Angle indices for novel-angle validation
        train_identity_ratio: Fraction of identities for training (e.g., 0.6 = 60%)
    
    Returns:
        Tuple of (train_data, val_novel_angle_data, val_novel_identity_data, statistics)
    """
    from .shapenet_downloader import scan_generated_stimuli
    
    # Load all stimuli
    all_stimulus_data = scan_generated_stimuli(stimuli_dir)
    
    if not all_stimulus_data:
        raise ValueError(f"No stimuli found in {stimuli_dir}")
    
    # Create splits
    splitter = ValidationSplitter(
        all_stimulus_data,
        train_angles=train_angles,
        val_angles=val_angles,
        train_identity_ratio=train_identity_ratio
    )
    
    train_data, val_novel_angle_data, val_novel_identity_data = splitter.create_splits()
    
    # Get statistics
    stats = splitter.get_split_statistics(train_data, val_novel_angle_data, val_novel_identity_data)
    
    return train_data, val_novel_angle_data, val_novel_identity_data, stats
