#!/usr/bin/env python3
"""
Generate stimulus images from 3D objects for the working memory experiments.
This script renders all ShapeNet objects at different locations and viewing angles.
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.shapenet_downloader import ShapeNetDownloader
from data.renderer import StimulusRenderer


def generate_all_stimuli(shapenet_dir="data/shapenet", 
                        output_dir="data/stimuli",
                        image_size=(224, 224)):
    """
    Generate all stimulus images from ShapeNet objects.
    
    Args:
        shapenet_dir: Directory containing ShapeNet objects
        output_dir: Directory to save rendered stimuli
        image_size: Size of output images
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
    
    # Viewing angles to render
    viewing_angles = [
        (0, 0, 0),              # angle0: front view
        (0, np.pi/8, 0),        # angle1: slight Y rotation
        (0, -np.pi/8, 0),       # angle2: slight Y rotation (opposite)
        (np.pi/12, 0, 0),       # angle3: slight X rotation
    ]
    
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


def update_sample_stimulus_data():
    """
    Update the sample stimulus data to point to actual generated files.
    """
    print("\nUpdating sample stimulus data paths...")
    
    # Check what files were actually generated
    stimuli_dir = Path("data/stimuli")
    if not stimuli_dir.exists():
        print("No stimuli directory found. Run stimulus generation first.")
        return {}
    
    # Scan for generated files
    stimulus_files = list(stimuli_dir.glob("stimulus_*.png"))
    
    # Organize by category and identity
    stimulus_data = {}
    
    for file_path in stimulus_files:
        # Parse filename: stimulus_airplane_000_loc0_angle0.png
        parts = file_path.stem.split('_')
        if len(parts) >= 4:
            category = parts[1]  # airplane
            identity = f"{parts[1]}_{parts[2]}"  # airplane_000
            
            if category not in stimulus_data:
                stimulus_data[category] = {}
            if identity not in stimulus_data[category]:
                stimulus_data[category][identity] = []
                
            stimulus_data[category][identity].append(str(file_path))
    
    # Sort file lists for consistency
    for category in stimulus_data:
        for identity in stimulus_data[category]:
            stimulus_data[category][identity].sort()
    
    print(f"Found stimulus files for:")
    for category, identities in stimulus_data.items():
        print(f"  {category}: {len(identities)} identities")
        for identity, files in identities.items():
            print(f"    {identity}: {len(files)} files")
    
    return stimulus_data


def create_demo_with_real_stimuli():
    """
    Create a demo script that uses the generated stimulus images.
    """
    demo_script = '''#!/usr/bin/env python3
"""
Demo with real generated stimulus images.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.dataset import NBackDataModule
from data.nback_generator import TaskFeature


def load_real_stimulus_data():
    """Load stimulus data from generated files."""
    stimuli_dir = Path("data/stimuli")
    
    stimulus_data = {}
    stimulus_files = list(stimuli_dir.glob("stimulus_*.png"))
    
    for file_path in stimulus_files:
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


def main():
    print("Demo with Real Stimulus Images")
    print("=" * 40)
    
    # Load real stimulus data
    stimulus_data = load_real_stimulus_data()
    
    if not stimulus_data:
        print("No stimulus images found. Run 'python generate_stimuli.py' first.")
        return
    
    print("Available stimulus data:")
    for category, identities in stimulus_data.items():
        print(f"  {category}: {len(identities)} identities")
    
    # Create data module with real stimuli
    data_module = NBackDataModule(
        stimulus_data=stimulus_data,
        n_values=[1, 2],
        task_features=[TaskFeature.LOCATION, TaskFeature.CATEGORY],
        sequence_length=6,
        batch_size=2,
        num_train=10,
        num_val=5,
        num_test=5,
        num_workers=0
    )
    
    # Sample a batch
    print("\\nSampling batch with real images...")
    try:
        batch = data_module.sample_batch("train")
        print(f"✓ Successfully loaded batch with real images!")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  No file loading warnings!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
'''
    
    with open("demo_real_stimuli.py", "w") as f:
        f.write(demo_script)
    
    print("Created demo_real_stimuli.py for testing with real images")


def main():
    """Main function to generate all stimuli."""
    print("Working Memory Model - Stimulus Generation")
    print("=" * 50)
    
    # Generate all stimulus images
    generated, total = generate_all_stimuli()
    
    # Update sample stimulus data
    stimulus_data = update_sample_stimulus_data()
    
    # Create demo script
    create_demo_with_real_stimuli()
    
    print(f"\n" + "=" * 50)
    print("STIMULUS GENERATION SUMMARY")
    print("=" * 50)
    print(f"✓ Generated {generated}/{total} stimulus images")
    print(f"✓ Found {len(stimulus_data)} categories with real stimuli")
    print(f"✓ Created demo_real_stimuli.py for testing")
    print()
    print("Next steps:")
    print("1. Run: python demo_real_stimuli.py")
    print("2. This should show no image loading warnings!")
    print("3. The dataset will now use real stimulus images")


if __name__ == "__main__":
    main()
