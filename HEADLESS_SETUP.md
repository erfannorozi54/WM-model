# Headless Server Setup for PyRender

This guide helps you set up 3D rendering on servers without displays (headless environments).

## The Problem

When running `python -m src.data.generate_stimuli` on a headless server, you may encounter:
```
pyglet.display.xlib.NoSuchDisplayException: Cannot connect to "None"
```

This happens because pyrender tries to use X11 display, which doesn't exist on headless servers.

## Solution: Use EGL (Hardware-Accelerated Headless Rendering)

The code now automatically detects headless environments and uses EGL. You just need to install system dependencies.

### Step 1: Install EGL System Libraries

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libgles2-mesa-dev
```

**CentOS/RHEL:**
```bash
sudo yum install -y mesa-libEGL mesa-libEGL-devel mesa-libGLES-devel
```

### Step 2: Verify Installation

Check if EGL is available:
```bash
ls /usr/lib/x86_64-linux-gnu/libEGL* 
# or
ls /usr/lib64/libEGL*
```

### Step 3: Run Your Code

The code will automatically detect the headless environment and use EGL:
```bash
python -m src.data.generate_stimuli
```

You should see:
```
Headless environment detected. Using EGL for OpenGL rendering.
Renderer mode: PyRender (3D)
```

## Alternative: OSMesa (Software Rendering)

If EGL is not available, you can use OSMesa (slower, CPU-based):

**Install OSMesa:**
```bash
# Ubuntu/Debian
sudo apt-get install -y libosmesa6 libosmesa6-dev

# CentOS/RHEL
sudo yum install -y mesa-libOSMesa mesa-libOSMesa-devel
```

**Set environment variable before running:**
```bash
export PYOPENGL_PLATFORM=osmesa
python -m src.data.generate_stimuli
```

## Verification

After setup, run a quick test:
```bash
python -c "import os; os.environ['PYOPENGL_PLATFORM'] = 'egl'; import pyrender; print('PyRender with EGL: OK')"
```

## GPU Servers

If you're on a GPU server with NVIDIA drivers:

1. Make sure EGL libraries are installed (see Step 1)
2. Verify NVIDIA EGL is available:
   ```bash
   ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia*
   ```
3. The code will automatically use GPU-accelerated EGL rendering

## Troubleshooting

### Still getting display errors?

Try explicitly setting the platform before running:
```bash
export PYOPENGL_PLATFORM=egl
python -m src.data.generate_stimuli
```

### EGL not found?

Install the development packages:
```bash
sudo apt-get install -y mesa-common-dev
```

### Want to use fallback rendering?

The code automatically falls back to matplotlib-based rendering if pyrender fails. While slower and less realistic, it still generates usable stimuli.

## Performance Comparison

- **EGL (Hardware)**: ~100-200 images/second
- **OSMesa (Software)**: ~10-20 images/second  
- **Matplotlib fallback**: ~5-10 images/second

For generating 320 stimuli, EGL is recommended for best performance.
