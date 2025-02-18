import os
import shutil

def create_project_structure():
    # Define the base directory
    base_dir = "e:\\Desktop\\docker\\dev\\cerebro\\zerolm"
    
    # Define the structure
    structure = {
        "zerolm": {
            "core": ["__init__.py", "model.py", "types.py"],
            "memory": ["__init__.py", "manager.py", "cache.py"],
            "vector": ["__init__.py", "manager.py", "similarity.py"],
            "utils": ["__init__.py", "tokenizer.py", "validation.py"],
            "context": ["__init__.py", "tracker.py", "weighting.py"],
            "learning": ["__init__.py", "validator.py", "patterns.py"],
        },
        "tests": {
            "core": ["__init__.py", "test_model.py"],
            "memory": ["__init__.py", "test_manager.py"],
            "vector": ["__init__.py", "test_similarity.py"],
            "utils": ["__init__.py", "test_tokenizer.py"],
        }
    }
    
    # Create directories and files
    for main_dir, subdirs in structure.items():
        for subdir, files in subdirs.items():
            dir_path = os.path.join(base_dir, main_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            
            for file in files:
                file_path = os.path.join(dir_path, file)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write('"""Module docstring."""\n\n')

    # Create main __init__.py
    with open(os.path.join(base_dir, "zerolm", "__init__.py"), "w") as f:
        f.write('"""ZeroLM - Training-Free Language Model."""\n\n')
        f.write('from .core.model import ZeroShotLM\n\n')
        f.write('__version__ = "0.1.0"\n')

if __name__ == "__main__":
    create_project_structure()