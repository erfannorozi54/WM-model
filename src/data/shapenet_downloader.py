"""
ShapeNet dataset downloader and organizer.
Downloads specific object categories needed for the working memory experiments.
"""

import os
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    hf_hub_download = None
    HAS_HF_HUB = False


# Shared constants for the data pipeline
DEFAULT_CATEGORIES = {
    "02691156": "airplane",
    "02958343": "car", 
    "03001627": "chair",
    "04379243": "table"
}

DEFAULT_VIEWING_ANGLES = [
    (0, 0, 0),              # angle0: front view (0°)
    (0, 0.785, 0),          # angle1: Y rotation (45° = π/4) - rotated right
    (0, -0.785, 0),         # angle2: Y rotation (-45° = -π/4) - rotated left
    (0.524, 0, 0),          # angle3: X rotation (30° = π/6) - tilted down
]


def scan_generated_stimuli(stimuli_dir: str = "data/stimuli") -> Dict[str, Dict[str, List[str]]]:
    """
    Scan stimuli directory and organize files by category and identity.
    
    Args:
        stimuli_dir: Directory containing generated stimuli
        
    Returns:
        Dict mapping {category: {identity: [file_paths]}}
    """
    stimuli_path = Path(stimuli_dir)
    if not stimuli_path.exists():
        return {}
    
    stimulus_data = {}
    stimulus_files = list(stimuli_path.glob("stimulus_*.png"))
    
    for file_path in stimulus_files:
        # Parse filename: stimulus_airplane_000_loc0_angle0.png
        parts = file_path.stem.split('_')
        if len(parts) >= 4:
            category = parts[1]
            identity = f"{parts[1]}_{parts[2]}"
            
            if category not in stimulus_data:
                stimulus_data[category] = {}
            if identity not in stimulus_data[category]:
                stimulus_data[category][identity] = []
            
            stimulus_data[category][identity].append(str(file_path))
    
    # Sort for consistency
    for category in stimulus_data:
        for identity in stimulus_data[category]:
            stimulus_data[category][identity].sort()
    
    return stimulus_data


def create_sample_stimulus_data() -> Dict[str, Dict[str, List[str]]]:
    """
    Create sample/placeholder stimulus data structure for testing.
    Matches the actual generated stimulus naming convention.
    
    Returns:
        Sample stimulus data mapping {category: {identity: [file_paths]}}
    """
    return {
        "airplane": {
            "airplane_000": [
                "data/stimuli/stimulus_airplane_000_loc0_angle0.png",
                "data/stimuli/stimulus_airplane_000_loc1_angle0.png",
                "data/stimuli/stimulus_airplane_000_loc2_angle0.png",
                "data/stimuli/stimulus_airplane_000_loc3_angle0.png"
            ],
            "airplane_001": [
                "data/stimuli/stimulus_airplane_001_loc0_angle0.png",
                "data/stimuli/stimulus_airplane_001_loc1_angle0.png",
                "data/stimuli/stimulus_airplane_001_loc2_angle0.png",
                "data/stimuli/stimulus_airplane_001_loc3_angle0.png"
            ]
        },
        "car": {
            "car_000": [
                "data/stimuli/stimulus_car_000_loc0_angle0.png",
                "data/stimuli/stimulus_car_000_loc1_angle0.png",
                "data/stimuli/stimulus_car_000_loc2_angle0.png", 
                "data/stimuli/stimulus_car_000_loc3_angle0.png"
            ],
            "car_001": [
                "data/stimuli/stimulus_car_001_loc0_angle0.png",
                "data/stimuli/stimulus_car_001_loc1_angle0.png",
                "data/stimuli/stimulus_car_001_loc2_angle0.png",
                "data/stimuli/stimulus_car_001_loc3_angle0.png"
            ]
        }
    }


class ShapeNetDownloader:
    """
    Downloads and organizes ShapeNet dataset for working memory experiments.
    
    Based on the paper's requirements:
    - 4 object categories
    - 2 identities per category
    - Organized for easy access during rendering
    """
    
    def __init__(self, data_dir: str = "data/shapenet", 
                 categories: Optional[Dict[str, str]] = None):
        """
        Initialize the ShapeNet downloader.
        
        Args:
            data_dir: Directory to store ShapeNet data
            categories: Dict of {category_id: category_name}
                       If None, uses DEFAULT_CATEGORIES
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Use default categories or provided ones
        self.categories = categories if categories is not None else DEFAULT_CATEGORIES.copy()
        self.objects_per_category = 2  # As specified in paper
        
    def generate_placeholder(self, category_id: str, category_name: str) -> bool:
        """
        Generate placeholder ShapeNet structure for testing.
        
        Args:
            category_id: ShapeNet category ID
            category_name: Human-readable category name
            
        Returns:
            bool: Success status
        """
        category_dir = self.data_dir / category_name
        category_dir.mkdir(exist_ok=True)
        
        print(f"Generating placeholder: {category_name} ({category_id})")
        
        # Create metadata file
        metadata = {
            "category_id": category_id,
            "category_name": category_name,
            "objects_count": self.objects_per_category,
            "type": "placeholder"
        }
        
        with open(category_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Create placeholder object directories
        for i in range(self.objects_per_category):
            obj_id = f"{category_id}_{i:03d}"
            obj_dir = category_dir / obj_id
            obj_dir.mkdir(exist_ok=True)
            
            # Placeholder .obj file
            placeholder_file = obj_dir / "model.obj"
            with open(placeholder_file, "w") as f:
                f.write(f"# Placeholder OBJ file for {category_name} object {i}\n")
                f.write("# Replace with actual ShapeNet .obj file\n")
                
        return True
    
    def generate_all_placeholders(self) -> Dict[str, bool]:
        """
        Generate placeholder structure for all categories.
        
        Returns:
            Dict mapping category names to success status
        """
        results = {}
        
        for category_id, category_name in self.categories.items():
            try:
                success = self.generate_placeholder(category_id, category_name)
                results[category_name] = success
                print(f"✓ {category_name}: {'Success' if success else 'Failed'}")
            except Exception as e:
                results[category_name] = False
                print(f"✗ {category_name}: Failed - {str(e)}")
                
        return results
    
    def download_from_huggingface(
        self,
        filename: str,
        repo_id: str = "ShapeNet/ShapeNetCore",
        token: Optional[str] = None,
        local_dir: Optional[Path] = None,
    ) -> Path:
        """
        Download ShapeNet archive from Hugging Face Hub.
        
        Args:
            filename: File to download (e.g., 'ShapeNetCore.v2.zip')
            repo_id: HF repository ID
            token: HF token (or use HUGGINGFACE_TOKEN env var)
            local_dir: Download directory
            
        Returns:
            Path to downloaded file
        """
        if not HAS_HF_HUB:
            raise RuntimeError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
        
        token = token or os.environ.get("HUGGINGFACE_TOKEN")
        download_dir = local_dir or (self.data_dir.parent / "downloads")
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {filename} from Hugging Face Hub...")
        print(f"Repository: {repo_id} (dataset)")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",  # ShapeNet is a dataset repository
            token=token,
            local_dir=str(download_dir),
        )
        
        print(f"✅ Downloaded to: {path}")
        return Path(path)
    
    def extract_archive(self, archive_path: Path, extract_to: Optional[Path] = None) -> Path:
        """
        Extract downloaded ShapeNet archive.
        
        Args:
            archive_path: Path to zip file
            extract_to: Extraction directory
            
        Returns:
            Path to extracted root directory
        """
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        extract_root = extract_to or (self.data_dir.parent / "raw")
        extract_root = Path(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {archive_path.name}...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_root)
        
        # Find extracted ShapeNet directory
        extracted = list(extract_root.glob("ShapeNetCore*"))
        result = extracted[0] if extracted else extract_root
        print(f"✅ Extracted to: {result}")
        return result
    
    def organize_real_shapenet(
        self,
        shapenet_root: Path,
        objects_per_category: Optional[int] = None,
    ) -> bool:
        """
        Organize downloaded ShapeNet data for experiments.
        
        Args:
            shapenet_root: Path to ShapeNetCore.v2 directory
            objects_per_category: Objects to select per category
            
        Returns:
            Success status
        """
        if objects_per_category is None:
            objects_per_category = self.objects_per_category
        
        shapenet_root = Path(shapenet_root)
        if not shapenet_root.exists():
            print(f"❌ ShapeNet root not found: {shapenet_root}")
            return False
        
        # Check for ShapeNetCore.v2 subdirectory
        if (shapenet_root / "ShapeNetCore.v2").exists():
            shapenet_root = shapenet_root / "ShapeNetCore.v2"
        
        print(f"Organizing ShapeNet from: {shapenet_root}")
        success_count = 0
        
        for category_id, category_name in self.categories.items():
            category_source = shapenet_root / category_id
            if not category_source.exists():
                print(f"⚠️  Category not found: {category_id}")
                continue
            
            # Get available objects
            object_dirs = [
                d for d in category_source.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            
            if not object_dirs:
                print(f"⚠️  No objects in {category_id}")
                continue
            
            # Create output directory
            category_output = self.data_dir / category_name
            category_output.mkdir(exist_ok=True)
            
            # Copy selected objects
            selected = object_dirs[:objects_per_category]
            for i, obj_dir in enumerate(selected):
                obj_name = f"{category_id}_{i:03d}"
                output_obj_dir = category_output / obj_name
                output_obj_dir.mkdir(exist_ok=True)
                
                # Find and copy .obj files
                obj_files = list(obj_dir.glob("**/*.obj"))
                if obj_files:
                    shutil.copy2(obj_files[0], output_obj_dir / "model.obj")
                    
                    # Copy .mtl files if present
                    for mtl in obj_dir.glob("**/*.mtl"):
                        shutil.copy2(mtl, output_obj_dir / mtl.name)
                    
                    print(f"✅ {category_name}: {obj_name}")
            
            success_count += 1
        
        print(f"\n✅ Organized {success_count}/{len(self.categories)} categories")
        return success_count == len(self.categories)
    
    def get_object_paths(self, category: str) -> List[Path]:
        """
        Get paths to all objects in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of paths to object files
        """
        category_dir = self.data_dir / category
        if not category_dir.exists():
            return []
            
        object_paths = []
        for obj_dir in category_dir.iterdir():
            if obj_dir.is_dir() and obj_dir.name != "metadata.json":
                obj_file = obj_dir / "model.obj"
                if obj_file.exists():
                    object_paths.append(obj_file)
                    
        return object_paths
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the downloaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            "categories": {},
            "total_objects": 0
        }
        
        for category_name in self.categories.values():
            objects = self.get_object_paths(category_name)
            info["categories"][category_name] = {
                "count": len(objects),
                "paths": [str(p) for p in objects]
            }
            info["total_objects"] += len(objects)
            
        return info


if __name__ == "__main__":
    # Example usage: generate placeholder data
    downloader = ShapeNetDownloader()
    results = downloader.generate_all_placeholders()
    
    print("\nDataset Info:", downloader.get_dataset_info())
