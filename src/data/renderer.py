"""
3D object renderer for generating 2D stimuli from ShapeNet objects.
Renders objects at specified locations on black background for working memory experiments.
"""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
import json
import os

# Optional imports for 3D rendering
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. 3D rendering will use fallback methods.")

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("Warning: pyrender not available. Using matplotlib-based rendering fallback.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available. Some 3D features may be limited.")


class StimulusRenderer:
    """
    Renders 3D objects into 2D images for working memory experiments.
    
    Features:
    - Renders objects at 4 specified screen locations
    - Black background
    - Consistent lighting and camera setup
    - Various viewing angles
    - Batch rendering capabilities
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 locations: Optional[List[Tuple[float, float]]] = None,
                 background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize the renderer.
        
        Args:
            image_size: Output image dimensions (width, height)
            locations: 4 locations for object placement as (x, y) coordinates
                      Normalized to [-1, 1] range
            background_color: RGB background color (0-1 range)
        """
        self.image_size = image_size
        self.background_color = background_color
        
        # Default 4 locations as used in working memory experiments
        if locations is None:
            self.locations = [
                (-0.5, 0.5),   # Top-left
                (0.5, 0.5),    # Top-right  
                (-0.5, -0.5),  # Bottom-left
                (0.5, -0.5)    # Bottom-right
            ]
        else:
            self.locations = locations
            
        # Setup rendering backend
        if PYRENDER_AVAILABLE:
            # Setup PyRender scene
            self.scene = pyrender.Scene(bg_color=list(background_color) + [1.0])
            
            # Camera setup
            self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            self.camera_pose = self._get_camera_pose()
            
            # Lighting setup
            self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            self.light_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            # Renderer
            self.renderer = pyrender.OffscreenRenderer(*image_size)
            self.use_fallback = False
        else:
            # Fallback to matplotlib-based rendering
            self.use_fallback = True
            print("Using fallback rendering (matplotlib-based)")
        
    def _get_camera_pose(self, distance: float = 3.0) -> np.ndarray:
        """Get camera pose matrix."""
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, distance],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    def load_mesh(self, obj_path: Union[str, Path]):
        """
        Load a 3D mesh from file.
        
        Args:
            obj_path: Path to .obj file
            
        Returns:
            Loaded mesh object (trimesh or fallback)
        """
        if TRIMESH_AVAILABLE:
            try:
                mesh = trimesh.load(str(obj_path))
                
                # Normalize mesh size
                mesh.apply_scale(1.0 / mesh.scale)
                
                # Center the mesh
                mesh.apply_translation(-mesh.centroid)
                
                return mesh
            except Exception as e:
                print(f"Warning: Could not load mesh {obj_path}: {e}")
                # Return a simple placeholder mesh
                return trimesh.creation.box(extents=[1, 1, 1])
        else:
            # Fallback: return a simple dictionary representing a box
            return {
                'type': 'box',
                'extents': [1, 1, 1],
                'vertices': np.array([
                    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                ])
            }
    
    def render_object(self, 
                     mesh,
                     location_idx: int,
                     rotation_angles: Tuple[float, float, float] = (0, 0, 0),
                     scale: float = 1.0) -> np.ndarray:
        """
        Render a single object at specified location.
        
        Args:
            mesh: 3D mesh to render (trimesh or fallback dict)
            location_idx: Index of location (0-3)
            rotation_angles: Rotation angles in radians (x, y, z)
            scale: Scale factor for the object
            
        Returns:
            Rendered image as numpy array (H, W, 3)
        """
        if location_idx >= len(self.locations):
            raise ValueError(f"Location index {location_idx} out of range")
        
        if self.use_fallback or not PYRENDER_AVAILABLE:
            return self._render_fallback(mesh, location_idx, rotation_angles, scale)
        
        # PyRender rendering
        # Clear scene
        self.scene.clear()
        
        # Create pyrender mesh
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5
        )
        
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        # Apply transformations
        transform = np.eye(4)
        
        # Scale
        transform[:3, :3] *= scale
        
        # Rotation
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler('xyz', rotation_angles)
        transform[:3, :3] = transform[:3, :3] @ rotation.as_matrix()
        
        # Translation to specified location
        loc_x, loc_y = self.locations[location_idx]
        transform[0, 3] = loc_x * 2.0  # Scale to world coordinates
        transform[1, 3] = loc_y * 2.0
        
        # Add to scene
        self.scene.add(py_mesh, pose=transform)
        self.scene.add(self.camera, pose=self.camera_pose)
        self.scene.add(self.light, pose=self.light_pose)
        
        # Render
        color, depth = self.renderer.render(self.scene)
        
        return color
    
    def _render_fallback(self, mesh, location_idx: int, 
                        rotation_angles: Tuple[float, float, float],
                        scale: float) -> np.ndarray:
        """
        Fallback rendering using matplotlib when pyrender is not available.
        
        Args:
            mesh: Mesh object (dict with 'vertices' key for fallback)
            location_idx: Location index
            rotation_angles: Rotation angles
            scale: Scale factor
            
        Returns:
            Rendered image as numpy array
        """
        # Create a simple 2D projection
        fig, ax = plt.subplots(1, 1, figsize=(self.image_size[0]/100, self.image_size[1]/100))
        fig.patch.set_facecolor('black')  # Set figure background to black
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_facecolor('black')
        ax.axis('off')
        
        # Get location
        loc_x, loc_y = self.locations[location_idx]
        
        # Simple shape rendering based on mesh type
        if isinstance(mesh, dict) and mesh.get('type') == 'box':
            # Render a simple rectangle
            rect_size = 0.2 * scale
            rect = plt.Rectangle(
                (loc_x - rect_size/2, loc_y - rect_size/2),
                rect_size, rect_size,
                facecolor='lightgray',
                edgecolor='white',
                linewidth=2
            )
            ax.add_patch(rect)
        else:
            # Fallback to a simple circle
            circle = plt.Circle((loc_x, loc_y), 0.1 * scale, 
                              facecolor='lightgray', edgecolor='white')
            ax.add_patch(circle)
        
        # Convert to image
        fig.canvas.draw()
        
        # Use buffer_rgba() for newer matplotlib versions
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]  # Remove alpha channel
        except AttributeError:
            # Fallback for older matplotlib versions
            try:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Final fallback - create a simple image
                buf = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        plt.close(fig)
        
        # Resize to target size
        from PIL import Image
        image = Image.fromarray(buf)
        image = image.resize(self.image_size)
        
        return np.array(image)
    
    def render_stimulus_set(self,
                           obj_paths: List[Union[str, Path]],
                           output_dir: Union[str, Path],
                           viewing_angles: Optional[List[Tuple[float, float, float]]] = None,
                           prefix: str = "stimulus") -> Dict[str, List[str]]:
        """
        Render a complete set of stimuli for all objects and locations.
        
        Args:
            obj_paths: List of paths to .obj files
            output_dir: Directory to save rendered images
            viewing_angles: List of rotation angles to render for each object
            prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping object names to lists of rendered image paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if viewing_angles is None:
            # Default viewing angles (slight rotations for variety)
            viewing_angles = [
                (0, 0, 0),
                (0, np.pi/8, 0),
                (0, -np.pi/8, 0),
                (np.pi/12, 0, 0)
            ]
        
        rendered_images = {}
        
        for obj_path in obj_paths:
            obj_path = Path(obj_path)
            obj_name = obj_path.stem
            
            print(f"Rendering {obj_name}...")
            
            # Load mesh
            mesh = self.load_mesh(obj_path)
            
            rendered_images[obj_name] = []
            
            # Render at each location and viewing angle
            for loc_idx in range(len(self.locations)):
                for angle_idx, angles in enumerate(viewing_angles):
                    # Render image
                    image = self.render_object(mesh, loc_idx, angles)
                    
                    # Save image
                    filename = f"{prefix}_{obj_name}_loc{loc_idx}_angle{angle_idx}.png"
                    filepath = output_dir / filename
                    
                    Image.fromarray(image).save(filepath)
                    rendered_images[obj_name].append(str(filepath))
                    
        return rendered_images
    
    def create_sample_stimulus(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a sample stimulus image for testing.
        
        Args:
            save_path: Optional path to save the sample image
            
        Returns:
            Sample stimulus image
        """
        # Create a simple box mesh
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        
        # Render at location 0
        image = self.render_object(mesh, location_idx=0)
        
        if save_path:
            Image.fromarray(image).save(save_path)
            print(f"Sample stimulus saved to {save_path}")
            
        return image
    
    def __del__(self):
        """Cleanup renderer resources."""
        if hasattr(self, 'renderer') and not self.use_fallback:
            try:
                self.renderer.delete()
            except:
                pass  # Ignore cleanup errors


def create_renderer_demo():
    """Create a demonstration of the renderer capabilities."""
    
    demo_script = '''#!/usr/bin/env python3
"""
Demo script for the stimulus renderer.
Creates sample stimuli to test the rendering pipeline.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.renderer import StimulusRenderer


def main():
    print("Stimulus Renderer Demo")
    print("=" * 30)
    
    # Create renderer
    renderer = StimulusRenderer(image_size=(224, 224))
    
    # Create sample stimuli
    output_dir = Path("data/sample_stimuli")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample stimulus...")
    sample_image = renderer.create_sample_stimulus(
        save_path=str(output_dir / "sample_stimulus.png")
    )
    
    print(f"Sample stimulus shape: {sample_image.shape}")
    print(f"Sample saved to: {output_dir / 'sample_stimulus.png'}")
    
    # Display locations
    print("\\nConfigured stimulus locations:")
    for i, (x, y) in enumerate(renderer.locations):
        print(f"  Location {i}: ({x:+.1f}, {y:+.1f})")
    
    print("\\nRenderer demo completed successfully!")


if __name__ == "__main__":
    main()
'''
    
    return demo_script


if __name__ == "__main__":
    # Create sample renderer
    renderer = StimulusRenderer()
    print("Renderer initialized successfully")
    print(f"Image size: {renderer.image_size}")
    print(f"Number of locations: {len(renderer.locations)}")
    
    # Create a sample stimulus
    sample = renderer.create_sample_stimulus("sample_output.png")
    print(f"Sample stimulus created with shape: {sample.shape}")
