#!/usr/bin/env python3
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
    
    print("\nDownload Summary:")
    print("-" * 20)
    for category, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {category}")
        
    # Print dataset info
    print("\nDataset Info:")
    print("-" * 20)
    info = downloader.get_dataset_info()
    print(f"Total categories: {len(info['categories'])}")
    print(f"Total objects: {info['total_objects']}")
    
    for category, data in info["categories"].items():
        print(f"  {category}: {data['count']} objects")


if __name__ == "__main__":
    main()
