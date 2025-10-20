#!/usr/bin/env python3
"""
Generate stimulus images from 3D objects for the working memory experiments.
This script renders all ShapeNet objects at different locations and viewing angles.

Usage:
    python -m src.data.generate_stimuli
    python -m src.data.generate_stimuli --shapenet-dir data/shapenet --output-dir data/stimuli
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Import from same package
from .shapenet_downloader import (
    ShapeNetDownloader, 
    DEFAULT_VIEWING_ANGLES,
    scan_generated_stimuli
)
from .renderer import StimulusRenderer


def generate_all_stimuli(shapenet_dir="data/shapenet", 
                        output_dir="data/stimuli",
                        image_size=(224, 224),
                        viewing_angles=None):
    """
    Generate all stimulus images from ShapeNet objects.
    
    Args:
        shapenet_dir: Directory containing ShapeNet objects
        output_dir: Directory to save rendered stimuli
        image_size: Size of output images
        viewing_angles: List of (x, y, z) rotation tuples. Uses DEFAULT_VIEWING_ANGLES if None
    """
    print("Generating Stimulus Images")
    print("=" * 40)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize downloader to get object information
    downloader = ShapeNetDownloader(data_dir=shapenet_dir)
    
    # Initialize renderer
    renderer = StimulusRenderer(image_size=image_size)
    
    print(f"Renderer mode: {'Fallback (matplotlib)' if renderer.use_fallback else 'PyRender (3D)'}")
    print(f"Output directory: {output_path}")
    print(f"Image size: {image_size}")
    
    # Use default viewing angles if not provided
    if viewing_angles is None:
        viewing_angles = DEFAULT_VIEWING_ANGLES
    
    total_images = 0
    generated_images = 0
    
    # Process each category
    for category_id, category_name in downloader.categories.items():
        print(f"\nProcessing category: {category_name}")
        
        # Get object paths for this category
        obj_paths = downloader.get_object_paths(category_name)
        
        if not obj_paths:
            print(f"  No objects found for {category_name}")
            continue
            
        # Process each object
        for obj_idx, obj_path in enumerate(obj_paths):
            obj_name = f"{category_name}_{obj_idx:03d}"
            print(f"  Rendering {obj_name}...")
            
            # Load the 3D mesh
            mesh = renderer.load_mesh(obj_path)
            
            # Render at each location and viewing angle
            for loc_idx in range(len(renderer.locations)):
                for angle_idx, angles in enumerate(viewing_angles):
                    
                    # Generate filename
                    filename = f"stimulus_{obj_name}_loc{loc_idx}_angle{angle_idx}.png"
                    filepath = output_path / filename
                    
                    total_images += 1
                    
                    # Skip if already exists
                    if filepath.exists():
                        generated_images += 1
                        continue
                    
                    try:
                        # Render the image
                        image = renderer.render_object(mesh, loc_idx, angles)
                        
                        # Save the image
                        from PIL import Image
                        Image.fromarray(image).save(filepath)
                        generated_images += 1
                        
                    except Exception as e:
                        print(f"    Warning: Failed to render {filename}: {e}")
    
    print(f"\nStimulus generation completed!")
    print(f"Generated: {generated_images}/{total_images} images")
    print(f"Output directory: {output_path}")
    
    return generated_images, total_images


def main():
    """Main function to generate all stimuli."""
    print("Working Memory Model - Stimulus Generation")
    print("=" * 50)
    
    # Generate all stimulus images
    generated, total = generate_all_stimuli()
    
    # Scan generated stimuli
    print("\nScanning generated stimuli...")
    stimulus_data = scan_generated_stimuli("data/stimuli")
    
    if stimulus_data:
        print(f"Found stimulus files for:")
        for category, identities in stimulus_data.items():
            print(f"  {category}: {len(identities)} identities")
            for identity, files in identities.items():
                print(f"    {identity}: {len(files)} files")
    
    print(f"\n" + "=" * 50)
    print("STIMULUS GENERATION SUMMARY")
    print("=" * 50)
    print(f"✓ Generated {generated}/{total} stimulus images")
    print(f"✓ Found {len(stimulus_data)} categories with real stimuli")
    print()
    print("Next steps:")
    print("1. Use the generated stimuli in your training pipeline")
    print("2. Run: python -m src.data.dataset (to test dataset loading)")


if __name__ == "__main__":
    main()
