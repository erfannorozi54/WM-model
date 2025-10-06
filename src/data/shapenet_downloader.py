"""
ShapeNet dataset downloader and organizer.
Downloads specific object categories needed for the working memory experiments.
"""

import os
import requests
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Optional
import shutil
from tqdm import tqdm


class ShapeNetDownloader:
    """
    Downloads and organizes ShapeNet dataset for working memory experiments.
    
    Based on the paper's requirements:
    - 4 object categories
    - 2 identities per category
    - Organized for easy access during rendering
    """
    
    def __init__(self, data_dir: str = "data/shapenet", 
                 categories: Optional[List[str]] = None):
        """
        Initialize the ShapeNet downloader.
        
        Args:
            data_dir: Directory to store ShapeNet data
            categories: List of ShapeNet category IDs to download
                       If None, uses default categories from paper
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Default categories based on common working memory experiments
        # These are ShapeNet category IDs
        if categories is None:
            self.categories = {
                "02691156": "airplane",
                "02958343": "car", 
                "03001627": "chair",
                "04379243": "table"
            }
        else:
            self.categories = {cat: cat for cat in categories}
            
        self.objects_per_category = 2  # As specified in paper
        
    def download_category(self, category_id: str, category_name: str) -> bool:
        """
        Download a specific category from ShapeNet.
        
        Args:
            category_id: ShapeNet category ID
            category_name: Human-readable category name
            
        Returns:
            bool: Success status
        """
        category_dir = self.data_dir / category_name
        category_dir.mkdir(exist_ok=True)
        
        print(f"Processing category: {category_name} ({category_id})")
        
        # For demonstration, we'll create placeholder structure
        # In practice, you would download from ShapeNet's official source
        # Note: ShapeNet requires registration and agreement to terms
        
        # Create metadata file for the category
        metadata = {
            "category_id": category_id,
            "category_name": category_name,
            "objects_count": self.objects_per_category,
            "download_date": "placeholder"
        }
        
        with open(category_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Create placeholder object directories
        for i in range(self.objects_per_category):
            obj_id = f"{category_id}_{i:03d}"
            obj_dir = category_dir / obj_id
            obj_dir.mkdir(exist_ok=True)
            
            # Placeholder for actual .obj files
            placeholder_file = obj_dir / "model.obj"
            with open(placeholder_file, "w") as f:
                f.write(f"# Placeholder OBJ file for {category_name} object {i}\n")
                f.write("# Replace with actual ShapeNet .obj file\n")
                
        return True
    
    def download_all_categories(self) -> Dict[str, bool]:
        """
        Download all specified categories.
        
        Returns:
            Dict mapping category names to download success status
        """
        results = {}
        
        for category_id, category_name in self.categories.items():
            try:
                success = self.download_category(category_id, category_name)
                results[category_name] = success
                print(f"✓ {category_name}: {'Success' if success else 'Failed'}")
            except Exception as e:
                results[category_name] = False
                print(f"✗ {category_name}: Failed - {str(e)}")
                
        return results
    
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


def create_download_script():
    """Create a standalone script for downloading ShapeNet data."""
    
    script_content = '''#!/usr/bin/env python3
"""
Standalone script to download ShapeNet dataset for working memory experiments.

IMPORTANT: This script creates placeholder files. To use real ShapeNet data:
1. Register at https://shapenet.org/
2. Download the ShapeNet Core v2 dataset
3. Replace the placeholder .obj files with actual ShapeNet models
4. Update the category IDs and object selections as needed

Usage:
    python download_shapenet.py [--data-dir DATA_DIR]
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data.shapenet_downloader import ShapeNetDownloader


def main():
    parser = argparse.ArgumentParser(description="Download ShapeNet dataset")
    parser.add_argument("--data-dir", default="data/shapenet", 
                       help="Directory to store ShapeNet data")
    args = parser.parse_args()
    
    print("ShapeNet Dataset Downloader")
    print("=" * 40)
    print("Note: This creates placeholder files for development.")
    print("For real data, register at https://shapenet.org/")
    print()
    
    downloader = ShapeNetDownloader(data_dir=args.data_dir)
    results = downloader.download_all_categories()
    
    print("\\nDownload Summary:")
    print("-" * 20)
    for category, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {category}")
        
    # Print dataset info
    print("\\nDataset Info:")
    print("-" * 20)
    info = downloader.get_dataset_info()
    print(f"Total categories: {len(info['categories'])}")
    print(f"Total objects: {info['total_objects']}")
    
    for category, data in info["categories"].items():
        print(f"  {category}: {data['count']} objects")


if __name__ == "__main__":
    main()
'''
    
    return script_content


if __name__ == "__main__":
    # Example usage
    downloader = ShapeNetDownloader()
    results = downloader.download_all_categories()
    
    print("\nDataset Info:", downloader.get_dataset_info())
