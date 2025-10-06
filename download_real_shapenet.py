#!/usr/bin/env python3
"""
Script to guide ShapeNet dataset download and organization for working memory experiments.

IMPORTANT: ShapeNet requires registration and manual download due to licensing.
This script provides guidance and automates the organization after download.
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm


class ShapeNetRealDownloader:
    """
    Guide for downloading and organizing real ShapeNet data.
    
    ShapeNet Core v2 contains high-quality 3D models organized by category.
    """
    
    def __init__(self, data_dir: str = "data/shapenet_real"):
        """
        Initialize the real ShapeNet downloader.
        
        Args:
            data_dir: Directory to organize ShapeNet data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Target categories for working memory experiments
        # These are real ShapeNet taxonomy IDs
        self.target_categories = {
            "02691156": "airplane",
            "02958343": "car", 
            "03001627": "chair",
            "04379243": "table"
        }
        
        # ShapeNet URLs and info
        self.shapenet_url = "https://shapenet.org/"
        self.shapenet_core_v2 = "https://shapenet.org/download/shapenetcore"
        
    def print_download_instructions(self):
        """Print detailed instructions for downloading ShapeNet."""
        print("=" * 70)
        print("SHAPENET CORE V2 DOWNLOAD INSTRUCTIONS")
        print("=" * 70)
        print()
        print("ShapeNet requires manual registration and download due to licensing.")
        print("Follow these steps:")
        print()
        print("STEP 1: Register for ShapeNet Access")
        print(f"   • Go to: {self.shapenet_url}")
        print("   • Click 'Sign Up' and create an account")
        print("   • Verify your email address")
        print("   • Academic/research affiliation may be required")
        print()
        print("STEP 2: Request ShapeNet Core v2 Access")
        print(f"   • Visit: {self.shapenet_core_v2}")
        print("   • Click 'Request Access'")
        print("   • Fill out the form with your research purpose")
        print("   • Wait for approval (usually 1-3 business days)")
        print()
        print("STEP 3: Download ShapeNet Core v2")
        print("   • Once approved, download the dataset")
        print("   • File: ShapeNetCore.v2.zip (~25GB)")
        print("   • Alternative: Download specific categories only")
        print()
        print("STEP 4: Run this script to organize the data")
        print("   • python download_real_shapenet.py --organize /path/to/ShapeNetCore.v2")
        print()
        print("TARGET CATEGORIES FOR WORKING MEMORY:")
        for cat_id, cat_name in self.target_categories.items():
            print(f"   • {cat_name}: {cat_id}")
        print()
    
    def organize_shapenet_data(self, shapenet_root: Path, 
                             objects_per_category: int = 2) -> bool:
        """
        Organize downloaded ShapeNet data for working memory experiments.
        
        Args:
            shapenet_root: Path to extracted ShapeNet root directory
            objects_per_category: Number of objects to select per category
            
        Returns:
            Success status
        """
        print("=" * 70)
        print("ORGANIZING SHAPENET DATA")
        print("=" * 70)
        
        if not shapenet_root.exists():
            print(f"❌ ShapeNet root not found: {shapenet_root}")
            return False
        
        # Look for ShapeNetCore.v2 structure
        core_path = shapenet_root / "ShapeNetCore.v2"
        if not core_path.exists():
            # Maybe the path is the core directory itself
            core_path = shapenet_root
        
        print(f"📁 ShapeNet root: {core_path}")
        
        success_count = 0
        
        for category_id, category_name in self.target_categories.items():
            print(f"\n🔍 Processing category: {category_name} ({category_id})")
            
            # Find category directory in ShapeNet
            category_source = core_path / category_id
            if not category_source.exists():
                print(f"   ⚠️  Category not found: {category_source}")
                continue
            
            # List available objects in this category
            object_dirs = [d for d in category_source.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')]
            
            if len(object_dirs) == 0:
                print(f"   ⚠️  No objects found in {category_source}")
                continue
            
            print(f"   📊 Found {len(object_dirs)} objects")
            
            # Select objects for working memory (take first N)
            selected_objects = object_dirs[:objects_per_category]
            
            # Create organized directory structure
            category_output = self.data_dir / category_name
            category_output.mkdir(exist_ok=True)
            
            for i, obj_dir in enumerate(selected_objects):
                obj_name = f"{category_id}_{i:03d}"
                output_obj_dir = category_output / obj_name
                output_obj_dir.mkdir(exist_ok=True)
                
                # Look for .obj file in the source
                obj_files = list(obj_dir.glob("**/*.obj"))
                
                if obj_files:
                    # Copy the first .obj file found
                    source_obj = obj_files[0]
                    target_obj = output_obj_dir / "model.obj"
                    
                    shutil.copy2(source_obj, target_obj)
                    print(f"   ✅ {obj_name}: {source_obj.name} -> {target_obj}")
                    
                    # Also copy any .mtl files (materials)
                    mtl_files = list(obj_dir.glob("**/*.mtl"))
                    for mtl_file in mtl_files:
                        target_mtl = output_obj_dir / mtl_file.name
                        shutil.copy2(mtl_file, target_mtl)
                        
                else:
                    print(f"   ❌ No .obj file found in {obj_dir}")
            
            success_count += 1
        
        print(f"\n📋 ORGANIZATION SUMMARY")
        print(f"   ✅ Successfully organized: {success_count}/{len(self.target_categories)} categories")
        
        return success_count == len(self.target_categories)
    
    def verify_organized_data(self) -> Dict:
        """Verify the organized ShapeNet data."""
        print("=" * 70)
        print("VERIFYING ORGANIZED DATA")
        print("=" * 70)
        
        verification = {
            "categories": {},
            "total_objects": 0,
            "valid_objects": 0
        }
        
        for category_name in self.target_categories.values():
            category_path = self.data_dir / category_name
            
            if not category_path.exists():
                verification["categories"][category_name] = {
                    "status": "missing",
                    "objects": 0
                }
                continue
            
            # Count objects in category
            object_dirs = [d for d in category_path.iterdir() if d.is_dir()]
            valid_objects = 0
            
            for obj_dir in object_dirs:
                obj_file = obj_dir / "model.obj"
                if obj_file.exists() and obj_file.stat().st_size > 100:  # At least 100 bytes
                    valid_objects += 1
            
            verification["categories"][category_name] = {
                "status": "found",
                "objects": len(object_dirs),
                "valid_objects": valid_objects
            }
            
            verification["total_objects"] += len(object_dirs)
            verification["valid_objects"] += valid_objects
            
        # Print verification results
        for category, info in verification["categories"].items():
            status_emoji = "✅" if info["status"] == "found" else "❌"
            print(f"{status_emoji} {category}: {info.get('valid_objects', 0)} valid objects")
        
        print(f"\n📊 TOTAL: {verification['valid_objects']}/{verification['total_objects']} valid objects")
        
        return verification
    
    def create_integration_script(self):
        """Create a script to integrate real ShapeNet data with the existing pipeline."""
        
        integration_script = f'''#!/usr/bin/env python3
"""
Integration script for real ShapeNet data with working memory pipeline.
Run this after organizing real ShapeNet data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.renderer import StimulusRenderer
from data.shapenet_downloader import ShapeNetDownloader


def integrate_real_shapenet():
    """Integrate real ShapeNet data with the existing pipeline."""
    print("Integrating Real ShapeNet Data")
    print("=" * 40)
    
    # Point to real ShapeNet data
    real_data_dir = "{self.data_dir}"
    
    # Create downloader pointing to real data
    downloader = ShapeNetDownloader(data_dir=real_data_dir)
    
    # Verify real data exists
    info = downloader.get_dataset_info()
    
    if info["total_objects"] == 0:
        print("❌ No real ShapeNet data found!")
        print("Run: python download_real_shapenet.py --organize /path/to/ShapeNetCore.v2")
        return False
    
    print(f"✅ Found {{info['total_objects']}} real ShapeNet objects")
    
    # Create renderer
    renderer = StimulusRenderer(image_size=(224, 224))
    
    # Generate stimuli from real objects
    print("\\nGenerating stimuli from real ShapeNet objects...")
    
    total_rendered = 0
    
    for category in downloader.categories.values():
        obj_paths = downloader.get_object_paths(category)
        
        if obj_paths:
            print(f"Rendering {{len(obj_paths)}} {{category}} objects...")
            
            rendered = renderer.render_stimulus_set(
                obj_paths=obj_paths,
                output_dir="data/stimuli_real",
                prefix="real_stimulus"
            )
            
            total_rendered += sum(len(paths) for paths in rendered.values())
    
    print(f"\\n✅ Generated {{total_rendered}} real stimulus images!")
    print("Real stimuli saved in: data/stimuli_real/")
    
    return True


if __name__ == "__main__":
    integrate_real_shapenet()
'''
        
        script_path = Path("integrate_real_shapenet.py")
        with open(script_path, "w") as f:
            f.write(integration_script)
        
        print(f"📝 Created integration script: {script_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and organize real ShapeNet data")
    parser.add_argument("--organize", type=str, 
                       help="Path to downloaded ShapeNet root directory")
    parser.add_argument("--verify", action="store_true",
                       help="Verify organized ShapeNet data")
    parser.add_argument("--objects-per-category", type=int, default=2,
                       help="Number of objects per category (default: 2)")
    
    args = parser.parse_args()
    
    downloader = ShapeNetRealDownloader()
    
    if args.organize:
        # Organize downloaded ShapeNet data
        shapenet_root = Path(args.organize)
        success = downloader.organize_shapenet_data(
            shapenet_root, 
            args.objects_per_category
        )
        
        if success:
            print("\n🎉 ShapeNet data organized successfully!")
            downloader.create_integration_script()
            print("\nNext steps:")
            print("1. Run: python integrate_real_shapenet.py")
            print("2. Use real stimuli in your experiments!")
        else:
            print("\n❌ Failed to organize ShapeNet data")
            return 1
            
    elif args.verify:
        # Verify organized data
        verification = downloader.verify_organized_data()
        
        if verification["valid_objects"] >= 8:  # 4 categories × 2 objects
            print("\n✅ Real ShapeNet data is ready!")
            return 0
        else:
            print("\n❌ Insufficient valid objects found")
            return 1
    
    else:
        # Show download instructions
        downloader.print_download_instructions()
        
        print("=" * 70)
        print("QUICK START COMMANDS")
        print("=" * 70)
        print("# After downloading ShapeNet:")
        print("python download_real_shapenet.py --organize /path/to/ShapeNetCore.v2")
        print()
        print("# Verify the organized data:")
        print("python download_real_shapenet.py --verify")
        print()
        print("# Integrate with existing pipeline:")
        print("python integrate_real_shapenet.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
