"""
Setup and Installation Script for SceneMotion-LLM
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_directories():
    """Create necessary project directories"""
    print("Creating project directories...")
    
    directories = [
        'dataset',
        'dataset/train',
        'dataset/test',
        'dataset/labels',
        'videos',
        'frames',
        'optical_flow',
        'checkpoints',
        'outputs',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    print("✓ All directories created\n")


def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("\n✓ Dependencies installed successfully\n")
        return True
    except Exception as e:
        print(f"\n✗ Error installing dependencies: {str(e)}")
        print("Try running: pip install -r requirements.txt\n")
        return False


def verify_installation():
    """Verify installation by checking imports"""
    print("Verifying installation...")
    
    packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'torchvision': 'TorchVision',
        'transformers': 'Transformers',
        'streamlit': 'Streamlit',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    installed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            installed.append(name)
            print(f"  ✓ {name}")
        except ImportError:
            missing.append((package, name))
            print(f"  ✗ {name}")
    
    if missing:
        print(f"\n⚠ Missing {len(missing)} package(s):")
        for package, name in missing:
            print(f"  - {name} (pip install {package})")
        return False
    else:
        print(f"\n✓ All packages verified\n")
        return True


def check_pytorch_gpu():
    """Check PyTorch GPU availability"""
    print("Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA Version: {torch.version.cuda}\n")
        else:
            print("  ⚠ CUDA not available (will use CPU)\n")
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {str(e)}\n")


def create_sample_files():
    """Create sample configuration files"""
    print("Creating sample files...")
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
checkpoints/
outputs/
frames/
optical_flow/
*.mp4
*.avi
*.mov

# Data
dataset/
videos/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Environment
.env
.env.local
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("  ✓ .gitignore created")
    
    # Create .env.example
    env_example = """
# PyTorch Device
DEVICE=cuda

# Model Configuration
MODEL_PATH=./checkpoints/best_model.pt

# Data Paths
DATASET_PATH=./datasets/CMU-MOSEI
FRAMES_PATH=./frames
OUTPUT_DIR=./outputs

# Training Parameters
BATCH_SIZE=8
LEARNING_RATE=1e-3
NUM_EPOCHS=50

# Logging
LOG_LEVEL=INFO
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example.strip())
    print("  ✓ .env.example created\n")


def main():
    """Main setup function"""
    print("="*70)
    print("SCENEMOTION-LLM: Setup and Installation")
    print("="*70)
    print()
    
    # Create directories
    setup_directories()
    
    # Create sample files
    create_sample_files()
    
    # Check GPU
    check_pytorch_gpu()
    
    # Install dependencies
    deps_ok = install_dependencies()
    
    # Verify installation
    verify_ok = verify_installation()
    
    # Summary
    print("="*70)
    if deps_ok and verify_ok:
        print("✓ SETUP COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nYou can now:")
        print("  1. Run examples: python examples.py")
        print("  2. Train model: python main_train.py --use-dummy-data")
        print("  3. Launch app: streamlit run app.py")
        print("  4. Run inference: python main_inference.py video.mp4")
    else:
        print("⚠ SETUP COMPLETED WITH ISSUES")
        print("="*70)
        print("\nPlease install missing packages and try again:")
        print("  pip install -r requirements.txt")
    
    print()


if __name__ == '__main__':
    main()
