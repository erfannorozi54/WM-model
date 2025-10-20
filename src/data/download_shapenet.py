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

    # Download via Hugging Face Hub
    python -m src.data.download_shapenet --download-hf ShapeNetCore.v2.zip --hf-token TOKEN

    # Organize manually downloaded ShapeNet
    python -m src.data.download_shapenet --organize /path/to/ShapeNetCore.v2

    # Verify organized data
    python -m src.data.download_shapenet --verify
"""

import sys
import argparse
from pathlib import Path

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

  # Download from Hugging Face Hub
  %(prog)s --download-hf ShapeNetCore.v2.zip --hf-token YOUR_TOKEN

  # Organize manually downloaded data
  %(prog)s --organize /path/to/ShapeNetCore.v2

  # Verify organized data
  %(prog)s --verify
        """
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--placeholder", action="store_true",
                     help="Generate placeholder data (for testing)")
    mode.add_argument("--download-hf", metavar="FILE",
                     help="Download from Hugging Face Hub (e.g., ShapeNetCore.v2.zip)")
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
                       help="HF token (or set HUGGINGFACE_TOKEN env var)")

    args = parser.parse_args()

    # Create downloader instance
    downloader = ShapeNetDownloader(data_dir=args.data_dir)

    # === PLACEHOLDER MODE ===
    if args.placeholder:
        print("=" * 70)
        print("GENERATING PLACEHOLDER SHAPENET DATA")
        print("=" * 70)
        
        results = downloader.generate_all_placeholders()
        
        success = sum(1 for v in results.values() if v)
        print(f"\n✅ Generated {success}/{len(results)} categories")
        print(f"📁 Location: {args.data_dir}\n")
        print("Next step: python scripts/data/generate_stimuli.py")
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

    # === DOWNLOAD FROM HUGGING FACE ===
    elif args.download_hf:
        try:
            # Download
            archive = downloader.download_from_huggingface(
                filename=args.download_hf,
                repo_id=args.hf_repo,
                token=args.hf_token,
            )
            
            # Extract
            extracted = downloader.extract_archive(archive)
            
            # Organize
            success = downloader.organize_real_shapenet(
                extracted,
                args.objects_per_category
            )
            
            if success:
                print("\n🎉 ShapeNet ready!")
                print(f"📁 Location: {args.data_dir}\n")
                print("Next step: python scripts/data/generate_stimuli.py")
                return 0
            else:
                print("\n❌ Organization failed")
                return 1
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
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
            print("Next step: python scripts/data/generate_stimuli.py")
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
        print("   python scripts/data/download_shapenet.py --placeholder\n")
        print("   • Instant generation, no downloads")
        print("   • Perfect for testing the pipeline\n")
        
        print("2. HUGGING FACE HUB (Automated Download)")
        print("   " + "-" * 66)
        print("   pip install huggingface_hub")
        print("   python scripts/data/download_shapenet.py \\")
        print("     --download-hf ShapeNetCore.v2.zip \\")
        print("     --hf-token YOUR_TOKEN\n")
        print("   • Requires HF account with ShapeNet access")
        print("   • ~25GB download\n")
        
        print("3. MANUAL DOWNLOAD (Pre-downloaded Data)")
        print("   " + "-" * 66)
        print("   # After downloading from https://shapenet.org/")
        print("   python scripts/data/download_shapenet.py \\")
        print("     --organize /path/to/ShapeNetCore.v2\n")
        
        print("=" * 70)
        print("For detailed help: python scripts/data/download_shapenet.py --help")
        print("=" * 70)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
