#!/usr/bin/env python3
"""
Environment setup script for the Working Memory Model project.
Handles virtual environment creation, dependency installation, and initial data setup.
"""

import subprocess
import sys
import os
from pathlib import Path
import venv


def run_command(command, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    if check and result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    
    return result


def create_virtual_environment(project_dir):
    """Create a Python virtual environment."""
    venv_dir = project_dir / "venv"
    
    if venv_dir.exists():
        print(f"Virtual environment already exists at {venv_dir}")
        return True
    
    print("Creating virtual environment...")
    try:
        venv.create(venv_dir, with_pip=True)
        print(f"✓ Virtual environment created at {venv_dir}")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False


def install_dependencies(project_dir):
    """Install project dependencies."""
    venv_dir = project_dir / "venv"
    requirements_file = project_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"✗ Requirements file not found: {requirements_file}")
        return False
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = venv_dir / "Scripts" / "pip"
    else:  # Unix/Linux/macOS
        pip_path = venv_dir / "bin" / "pip"
    
    print("Installing dependencies...")
    
    # Upgrade pip first
    result = run_command(f"{pip_path} install --upgrade pip", check=False)
    
    # Install requirements
    result = run_command(f"{pip_path} install -r {requirements_file}")
    
    if result:
        print("✓ Dependencies installed successfully")
        return True
    else:
        print("✗ Failed to install dependencies")
        return False


def setup_project_structure(project_dir):
    """Ensure all necessary directories exist."""
    directories = [
        "data",
        "data/shapenet",
        "data/stimuli", 
        "data/sample_stimuli",
        "models",
        "outputs",
        "logs",
        "notebooks"
    ]
    
    print("Setting up project directories...")
    for dir_name in directories:
        dir_path = project_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_name}/")
    
    return True


def create_activation_scripts(project_dir):
    """Create convenience scripts for environment activation."""
    venv_dir = project_dir / "venv"
    
    # Unix/Linux/macOS activation script
    activate_script = project_dir / "activate_env.sh"
    with open(activate_script, "w") as f:
        f.write(f"""#!/bin/bash
# Activation script for Working Memory Model project

source {venv_dir}/bin/activate
export PYTHONPATH="${project_dir}/src:$PYTHONPATH"
echo "Working Memory Model environment activated!"
echo "Project directory: {project_dir}"
echo "Python path includes: {project_dir}/src"
""")
    
    # Make executable
    os.chmod(activate_script, 0o755)
    
    # Windows activation script
    activate_bat = project_dir / "activate_env.bat"
    with open(activate_bat, "w") as f:
        f.write(f"""@echo off
REM Activation script for Working Memory Model project

call {venv_dir}\\Scripts\\activate.bat
set PYTHONPATH={project_dir}\\src;%PYTHONPATH%
echo Working Memory Model environment activated!
echo Project directory: {project_dir}
echo Python path includes: {project_dir}\\src
""")
    
    print("✓ Environment activation scripts created")
    print(f"  - Unix/Linux/macOS: {activate_script}")
    print(f"  - Windows: {activate_bat}")
    
    return True


def create_readme(project_dir):
    """Create a comprehensive README file."""
    readme_content = """# Working Memory Model

A PyTorch implementation of a neural network model for working memory experiments using N-back tasks with 3D object stimuli.

## Project Structure

```
WM-model/
├── src/                    # Source code
│   ├── data/              # Data handling modules
│   │   ├── shapenet_downloader.py  # ShapeNet dataset management
│   │   ├── renderer.py            # 3D object rendering
│   │   ├── nback_generator.py     # N-back trial generation
│   │   └── dataset.py             # PyTorch Dataset/DataLoader
│   ├── models/            # Neural network models
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   ├── shapenet/         # ShapeNet 3D objects
│   ├── stimuli/          # Rendered 2D stimuli
│   └── sample_stimuli/   # Sample/demo stimuli
├── notebooks/            # Jupyter notebooks
├── models/               # Trained model checkpoints
├── outputs/              # Experiment outputs
└── logs/                 # Training logs

```

## Quick Start

### 1. Environment Setup

Activate the environment:
```bash
# Unix/Linux/macOS
source activate_env.sh

# Windows
activate_env.bat

# Or manually:
source venv/bin/activate  # Unix/Linux/macOS
# venv\\Scripts\\activate.bat  # Windows
export PYTHONPATH="src:$PYTHONPATH"
```

### 2. Download and Setup Data

```python
# Download ShapeNet data (placeholder setup)
python -m src.data.shapenet_downloader

# Test the renderer
python -m src.data.renderer

# Test the N-back generator
python -m src.data.nback_generator

# Test the full data pipeline
python test_data_pipeline.py
```

### 3. Generate Stimuli

```python
from src.data.renderer import StimulusRenderer
from src.data.shapenet_downloader import ShapeNetDownloader

# Setup data
downloader = ShapeNetDownloader()
downloader.download_all_categories()

# Render stimuli
renderer = StimulusRenderer()
# Add rendering code here...
```

### 4. Create N-back Dataset

```python
from src.data.dataset import NBackDataModule
from src.data.nback_generator import TaskFeature

# Create data module
data_module = NBackDataModule(
    stimulus_data=your_stimulus_data,
    n_values=[1, 2, 3],
    task_features=[TaskFeature.LOCATION, TaskFeature.IDENTITY, TaskFeature.CATEGORY],
    batch_size=32
)

# Get data loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

## Components

### Data Pipeline

1. **ShapeNet Downloader** (`src/data/shapenet_downloader.py`)
   - Downloads and organizes 3D object data
   - 4 categories, 2 identities each (as per paper)

2. **Stimulus Renderer** (`src/data/renderer.py`) 
   - Renders 3D objects to 2D images
   - 4 screen locations, black background
   - Multiple viewing angles

3. **N-back Generator** (`src/data/nback_generator.py`)
   - Generates trial sequences for Location, Identity, Category tasks
   - Configurable N (1-back, 2-back, 3-back)
   - Flexible sequence length and match probability

4. **PyTorch Dataset** (`src/data/dataset.py`)
   - Wraps generators in PyTorch Dataset/DataLoader
   - Handles batching, image loading, preprocessing
   - Train/validation/test splits

### Key Features

- **Flexible N-back Tasks**: Support for 1-back, 2-back, 3-back conditions
- **Multiple Task Types**: Location, Identity, and Category matching
- **3D Stimulus Rendering**: From ShapeNet objects to 2D images
- **PyTorch Integration**: Full DataLoader support with transforms
- **Configurable Parameters**: Sequence length, match probability, batch size

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- PyTorch >= 2.0.0
- PyTorch3D >= 0.7.5 (for 3D rendering)
- scikit-learn >= 1.3.0 (for analyses)
- trimesh, pyrender (3D processing)

## Development

### Running Tests

```bash
python test_data_pipeline.py
```

### Creating Custom Experiments

```python
# See notebooks/ directory for examples
# Customize n_values, task_features, and other parameters as needed
```

## Notes

- This implementation creates placeholder ShapeNet files for development
- For real experiments, download actual ShapeNet data from https://shapenet.org/
- The renderer can be extended for additional stimulus variations
- N-back parameters can be adjusted for different experimental designs

## Citation

If you use this code, please cite the original working memory paper that inspired this implementation.
"""
    
    readme_path = project_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print("✓ README.md created")
    return True


def main():
    """Main setup function."""
    print("Working Memory Model - Environment Setup")
    print("=" * 50)
    
    # Get project directory
    project_dir = Path(__file__).parent.absolute()
    print(f"Project directory: {project_dir}")
    
    success = True
    
    # Step 1: Create virtual environment
    if not create_virtual_environment(project_dir):
        success = False
    
    # Step 2: Install dependencies
    if success and not install_dependencies(project_dir):
        success = False
    
    # Step 3: Setup project structure
    if success and not setup_project_structure(project_dir):
        success = False
    
    # Step 4: Create activation scripts
    if success and not create_activation_scripts(project_dir):
        success = False
    
    # Step 5: Create README
    if success and not create_readme(project_dir):
        success = False
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("✓ Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate the environment:")
        print("   source activate_env.sh  # Unix/Linux/macOS")
        print("   activate_env.bat        # Windows")
        print("2. Test the setup:")
        print("   python test_data_pipeline.py")
        print("3. Start developing!")
    else:
        print("✗ Environment setup failed!")
        print("Please check the error messages above and try again.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
