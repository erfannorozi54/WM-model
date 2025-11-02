#!/usr/bin/env python3
"""
ShapeNet dataset download and organization CLI.

Unified interface for all ShapeNet download modes:
- Placeholder generation (for testing)
- Hugging Face Hub download (automated)
- Manual organization (for pre-downloaded data)

Usage:
    # Generate placeholder data
    python -m src.data.download_shapenet --placeholder

    # Download categories from HuggingFace (if you have access)
    # Minimal mode - only airplane and car
    python -m src.data.download_shapenet --download-categories --minimal
    
    # Download all 4 categories
    python -m src.data.download_shapenet --download-categories
    
    # Download single category by ID (02691156 = airplane)
    python -m src.data.download_shapenet --download-hf 02691156.zip

    # Organize manually downloaded ShapeNet
    python -m src.data.download_shapenet --organize /path/to/ShapeNetCore.v2

    # Verify organized data
    python -m src.data.download_shapenet --verify

Note: 
- ShapeNet is a GATED dataset. Request access at: https://huggingface.co/datasets/ShapeNet/ShapeNetCore
- Create a .env file with HUGGINGFACE_TOKEN=your_token_here (see .env.example)
- Manual download is recommended due to large file size and access requirements
"""

import sys
import argparse
import os
import shutil
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Continuing without .env file support...\n")

# Import from same package
from .shapenet_downloader import ShapeNetDownloader


def main():
    parser = argparse.ArgumentParser(
        description="ShapeNet dataset download and organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with placeholder data
  %(prog)s --placeholder

  # Download categories from HuggingFace (RECOMMENDED if you have access)
  # Minimal mode - only airplane and car (faster)
  %(prog)s --download-categories --minimal
  
  # Download all 4 categories
  %(prog)s --download-categories
  
  # Download specific categories
  %(prog)s --download-categories --categories airplane chair
  
  # Download single file by category ID (02691156 = airplane)
  %(prog)s --download-hf 02691156.zip

  # Organize manually downloaded data
  %(prog)s --organize /path/to/ShapeNetCore.v2

  # Verify organized data
  %(prog)s --verify

Category IDs:
  02691156 = airplane (3.4 GB)
  02958343 = car (file size varies)
  03001627 = chair (file size varies)  
  04379243 = table (file size varies)

Note: Create a .env file with HUGGINGFACE_TOKEN=your_token (see .env.example)
        """
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--placeholder", action="store_true",
                     help="Generate placeholder data (for testing)")
    mode.add_argument("--download-hf", metavar="FILE",
                     help="Download from Hugging Face Hub (e.g., 02691156.zip for airplane)")
    mode.add_argument("--download-categories", action="store_true",
                     help="Download category files from HuggingFace (use with --categories or --minimal)")
    mode.add_argument("--organize", metavar="PATH",
                     help="Organize pre-downloaded ShapeNet from PATH")
    mode.add_argument("--verify", action="store_true",
                     help="Verify organized data")

    # Configuration
    parser.add_argument("--data-dir", default="data/shapenet",
                       help="Output directory (default: data/shapenet)")
    parser.add_argument("--objects-per-category", type=int, default=2,
                       help="Objects per category (default: 2)")

    # Hugging Face options
    parser.add_argument("--hf-repo", default="ShapeNet/ShapeNetCore",
                       help="HF repository ID (default: ShapeNet/ShapeNetCore)")
    parser.add_argument("--hf-token",
                       help="HF token (or set HUGGINGFACE_TOKEN in .env file)")
    
    # Category selection (mutually exclusive)
    category_group = parser.add_mutually_exclusive_group()
    category_group.add_argument("--categories", nargs="+",
                       help="Specific categories to download (e.g., airplane car chair table)")
    category_group.add_argument("--minimal", action="store_true",
                       help="Download only minimal essential categories (airplane, car) - saves space and time")

    args = parser.parse_args()

    # Get HuggingFace token from args or environment
    hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")
    
    if (args.download_hf or args.download_categories) and not hf_token:
        print("❌ Error: HuggingFace token required for download.")
        print("   Either:")
        print("   1. Add HUGGINGFACE_TOKEN to your .env file")
        print("   2. Use --hf-token argument")
        print("\n⚠️  Note: ShapeNet is a GATED dataset.")
        print("   Request access at: https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
        return 1

    # Create downloader instance with optional category filter
    from .shapenet_downloader import DEFAULT_CATEGORIES
    
    categories = None
    if args.minimal:
        # Minimal mode: only essential categories
        minimal_cats = ["airplane", "car"]
        categories = {k: v for k, v in DEFAULT_CATEGORIES.items() if v in minimal_cats}
        print(f"🎯 Minimal mode: Selected {', '.join(categories.values())}")
    elif args.categories:
        # Filter categories based on user selection
        categories = {k: v for k, v in DEFAULT_CATEGORIES.items() if v in args.categories}
        if not categories:
            print(f"❌ Error: No matching categories found.")
            print(f"   Available: {', '.join(DEFAULT_CATEGORIES.values())}")
            return 1
        print(f"✅ Selected categories: {', '.join(categories.values())}")
    
    downloader = ShapeNetDownloader(data_dir=args.data_dir, categories=categories)

    # === PLACEHOLDER MODE ===
    if args.placeholder:
        print("=" * 70)
        print("GENERATING PLACEHOLDER SHAPENET DATA")
        print("=" * 70)
        
        results = downloader.generate_all_placeholders()
        
        success = sum(1 for v in results.values() if v)
        print(f"\n✅ Generated {success}/{len(results)} categories")
        print(f"📁 Location: {args.data_dir}\n")
        print("Next step: python -m src.data.generate_stimuli")
        return 0

    # === VERIFY MODE ===
    elif args.verify:
        print("=" * 70)
        print("VERIFYING SHAPENET DATA")
        print("=" * 70)
        
        info = downloader.get_dataset_info()
        
        for category, data in info["categories"].items():
            count = data["count"]
            status = "✅" if count >= 2 else "⚠️ "
            print(f"{status} {category}: {count} objects")
        
        total = info["total_objects"]
        print(f"\n📊 Total: {total} objects")
        
        if total >= 8:
            print("\n✅ Dataset ready!")
            return 0
        else:
            print("\n⚠️  Insufficient objects (need at least 8)")
            return 1

    # === DOWNLOAD CATEGORIES FROM HUGGING FACE ===
    elif args.download_categories:
        try:
            print("=" * 70)
            print("DOWNLOADING SHAPENET CATEGORIES FROM HUGGING FACE")
            print("=" * 70)
            
            # Get category IDs to download
            categories_to_download = categories if categories else DEFAULT_CATEGORIES
            
            print(f"\nCategories to download: {', '.join(categories_to_download.values())}")
            print(f"Total files: {len(categories_to_download)}\n")
            
            downloaded_files = []
            for cat_id, cat_name in categories_to_download.items():
                filename = f"{cat_id}.zip"
                print(f"📥 Downloading {cat_name} ({filename})...")
                
                try:
                    archive = downloader.download_from_huggingface(
                        filename=filename,
                        repo_id=args.hf_repo,
                        token=hf_token,
                    )
                    downloaded_files.append((archive, cat_id, cat_name))
                    print(f"✅ Downloaded {cat_name}\n")
                except Exception as e:
                    print(f"⚠️  Failed to download {cat_name}: {e}\n")
                    continue
            
            if not downloaded_files:
                print("\n❌ No files were downloaded successfully")
                return 1
            
            # Extract and organize each category
            print("\n" + "=" * 70)
            print("EXTRACTING AND ORGANIZING")
            print("=" * 70 + "\n")
            
            for archive, cat_id, cat_name in downloaded_files:
                print(f"📦 Processing {cat_name}...")
                extracted = downloader.extract_archive(archive)
                
                # Organize this category
                success = downloader.organize_real_shapenet(
                    extracted,
                    args.objects_per_category
                )
                
                if success:
                    print(f"✅ {cat_name} organized")
                else:
                    print(f"⚠️  {cat_name} organization had issues")
                
                # Clean up extracted files (keep the zip)
                try:
                    if extracted.exists() and extracted.is_dir():
                        shutil.rmtree(extracted)
                        print(f"🧹 Cleaned up extracted files for {cat_name}\n")
                except Exception as e:
                    print(f"⚠️  Could not clean up {extracted}: {e}\n")
            
            print("\n🎉 Download and organization complete!")
            print(f"📁 Location: {args.data_dir}\n")
            print("Next step: python -m src.data.generate_stimuli")
            return 0
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1

    # === DOWNLOAD FROM HUGGING FACE ===
    elif args.download_hf:
        try:
            # Download
            archive = downloader.download_from_huggingface(
                filename=args.download_hf,
                repo_id=args.hf_repo,
                token=hf_token,
            )
            
            # Extract
            extracted = downloader.extract_archive(archive)
            
            # Organize
            success = downloader.organize_real_shapenet(
                extracted,
                args.objects_per_category
            )
            
            # Clean up extracted files (keep the zip)
            try:
                if extracted.exists() and extracted.is_dir():
                    shutil.rmtree(extracted)
                    print(f"🧹 Cleaned up extracted files")
            except Exception as e:
                print(f"⚠️  Could not clean up {extracted}: {e}")
            
            if success:
                print("\n🎉 ShapeNet ready!")
                print(f"📁 Location: {args.data_dir}\n")
                print("Next step: python -m src.data.generate_stimuli")
                return 0
            else:
                print("\n❌ Organization failed")
                return 1
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if "404" in str(e) or "Repository Not Found" in str(e):
                print("\n💡 Possible causes:")
                print("   1. ShapeNet is a GATED dataset - you may not have access")
                print("   2. Request access at: https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
                print("   3. The filename may not exist in the repository")
                print("   4. Check available files after getting access")
                print("\n💡 Alternative: Use manual download instead")
                print("   1. Download from https://shapenet.org/ or HuggingFace")
                print("   2. Use --organize to process the downloaded files")
            return 1

    # === ORGANIZE MANUAL DOWNLOAD ===
    elif args.organize:
        success = downloader.organize_real_shapenet(
            Path(args.organize),
            args.objects_per_category
        )
        
        if success:
            print("\n🎉 ShapeNet organized!")
            print(f"📁 Location: {args.data_dir}\n")
            print("Next step: python -m src.data.generate_stimuli")
            return 0
        else:
            print("\n❌ Organization failed")
            return 1

    # === SHOW HELP ===
    else:
        print("=" * 70)
        print("SHAPENET DATASET DOWNLOAD")
        print("=" * 70)
        print("\nChoose one of three options:\n")
        
        print("1. QUICK START (Placeholder Data)")
        print("   " + "-" * 66)
        print("   python -m src.data.download_shapenet --placeholder\n")
        print("   • Instant generation, no downloads")
        print("   • Perfect for testing the pipeline\n")
        
        print("2. HUGGING FACE HUB (Automated Download)")
        print("   " + "-" * 66)
        print("   ✅ If you have access, download by category:\n")
        print("   # Minimal mode (airplane + car only)")
        print("   python -m src.data.download_shapenet \\")
        print("     --download-categories --minimal\n")
        print("   # All 4 categories")
        print("   python -m src.data.download_shapenet \\")
        print("     --download-categories\n")
        print("   # Single category (02691156 = airplane)")
        print("   python -m src.data.download_shapenet \\")
        print("     --download-hf 02691156.zip\n")
        print("   • Requires HF access (request at:)")
        print("     https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
        print("   • Token auto-loaded from .env file")
        print("   • Downloads only needed categories\n")
        
        print("3. MANUAL DOWNLOAD (Pre-downloaded Data)")
        print("   " + "-" * 66)
        print("   # After downloading from https://shapenet.org/")
        print("   python -m src.data.download_shapenet \\")
        print("     --organize /path/to/ShapeNetCore.v2\n")
        
        print("=" * 70)
        print("For detailed help: python -m src.data.download_shapenet --help")
        print("=" * 70)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
