#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main interface script that orchestrates the entire pipeline.
Running this script will execute all necessary components in the proper sequence.
"""

import os
import sys
import argparse
import logging
import importlib
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ManeProject")

def setup_environment():
    """Set up the necessary environment for running the code."""
    logger.info("Setting up environment...")
    
    # Get the repository root directory
    repo_root = Path(__file__).absolute().parent
    
    # Add repository root to system path for importing modules
    sys.path.insert(0, str(repo_root))
    
    # Check for requirements.txt and install if needed
    requirements_path = repo_root / "requirements.txt"
    if not requirements_path.exists():
        logger.warning("requirements.txt not found. Creating basic requirements file...")
        with open(requirements_path, "w") as f:
            f.write("""torch>=1.10.0
torchvision>=0.11.0
numpy>=1.20.0
scipy>=1.7.0
Pillow>=8.3.0
opencv-python>=4.5.0
pandas>=1.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
scikit-learn>=1.0.0
""")
    
    logger.info("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
    except subprocess.CalledProcessError:
        logger.error("Failed to install requirements. Please install them manually.")
        logger.info(f"Run: pip install -r {requirements_path}")
    
    return repo_root

def discover_modules(repo_root):
    """Discover and return all Python modules in the repository."""
    modules = {}
    
    # Common module names to look for in an ML project
    essential_modules = [
        "data_preparation", "preprocess", "preprocessing", 
        "model", "models", "train", "trainer", "training",
        "evaluate", "evaluation", "predict", "prediction"
    ]
    
    # Search for Python files
    for path in repo_root.glob("**/*.py"):
        if path.name == "__init__.py" or path.name == "interface.py":
            continue
            
        relative_path = path.relative_to(repo_root)
        module_path = str(relative_path.with_suffix("")).replace(os.sep, ".")
        
        # Determine if the module is essential and its probable execution order
        priority = float('inf')
        for i, name in enumerate(essential_modules):
            if name in module_path:
                priority = i
                break
        
        modules[module_path] = {
            "path": path,
            "priority": priority,
            "is_essential": priority < float('inf')
        }
    
    return modules

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the complete pipeline for the Mane project")
    
    # Add general arguments
    parser.add_argument("--input", "-i", type=str, help="Path to input data")
    parser.add_argument("--output", "-o", type=str, help="Path to save output")
    parser.add_argument("--model", "-m", type=str, help="Path to model weights")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--skip_training", action="store_true", help="Skip training phase")
    parser.add_argument("--only_inference", action="store_true", help="Run only inference")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()

def run_module(module_name, module_info, args):
    """Run a specific module with the given arguments."""
    logger.info(f"Running module: {module_name}")
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Look for standard entry points
        entry_points = ["main", "run", "execute", "__main__"]
        
        for entry_point in entry_points:
            if hasattr(module, entry_point) and callable(getattr(module, entry_point)):
                function = getattr(module, entry_point)
                try:
                    function(args)
                    return True
                except TypeError:
                    # Try without arguments
                    try:
                        function()
                        return True
                    except Exception as e:
                        logger.error(f"Error running {module_name}.{entry_point}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error running {module_name}.{entry_point}: {str(e)}")
        
        # If no entry point was found or executed successfully
        logger.warning(f"No valid entry point found in {module_name}")
        return False
        
    except ImportError as e:
        logger.error(f"Failed to import module {module_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in module {module_name}: {str(e)}")
        return False

def main():
    """Main function to run the entire pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup environment
    repo_root = setup_environment()
    
    # Discover modules
    modules = discover_modules(repo_root)
    
    # Sort modules by priority
    sorted_modules = sorted(
        [(name, info) for name, info in modules.items()],
        key=lambda x: x[1]["priority"]
    )
    
    # Run modules based on flags
    success_count = 0
    total_modules = 0
    
    for module_name, module_info in sorted_modules:
        # Skip training if requested
        if args.skip_training and ("train" in module_name or "training" in module_name):
            logger.info(f"Skipping training module: {module_name}")
            continue
            
        # Run only inference modules if requested
        if args.only_inference and not any(x in module_name for x in ["predict", "inference", "evaluation"]):
            logger.info(f"Skipping non-inference module: {module_name}")
            continue
            
        total_modules += 1
        if run_module(module_name, module_info, args):
            success_count += 1
    
    logger.info(f"Execution completed. {success_count}/{total_modules} modules executed successfully.")

if __name__ == "__main__":
    main()
