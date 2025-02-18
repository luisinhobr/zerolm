import os
from pathlib import Path

def verify_distribution():
    base_dir = Path("e:/Desktop/docker/dev/cerebro/zerolm")
    
    # Define expected classes and their locations
    expected_classes = {
        "core/types.py": ["ResponseType", "Response", "ValidationResult", "MemoryStats"],
        "core/model.py": ["ZeroShotLM"],
        "memory/cache.py": ["ConcurrentLRUCache"],
        "memory/manager.py": ["AdaptiveMemoryManager"],
        "vector/manager.py": ["VectorManager"],
        "vector/similarity.py": ["PatternMatcher"],
        "context/tracker.py": ["HierarchicalContext"],
        "context/weighting.py": ["TemporalWeightingSystem", "ContextWeighter"],
        "learning/validator.py": ["LearningValidator"],
        "learning/patterns.py": ["AutoCorrector", "TemplateEnforcer", "TemplateValidator"],
        "utils/tokenizer.py": ["Tokenizer"],
        "utils/validation.py": ["ValidationUtils"]
    }
    
    # Read original file content
    with open(base_dir / "zerolm.py", "r", encoding="utf-8") as f:
        original_content = f.read()
    
    missing_classes = []
    empty_files = []
    
    # Check each expected file and its classes
    for rel_path, classes in expected_classes.items():
        file_path = base_dir / "zerolm" / rel_path
        
        if not file_path.exists():
            missing_classes.extend(classes)
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if len(content.strip()) < 50:  # Arbitrary minimum content length
            empty_files.append(rel_path)
            
        for class_name in classes:
            if f"class {class_name}" not in content:
                missing_classes.append(class_name)
    
    return missing_classes, empty_files

if __name__ == "__main__":
    missing, empty = verify_distribution()
    
    if missing or empty:
        print("Distribution verification failed!")
        
        if missing:
            print("\nMissing classes:")
            for class_name in missing:
                print(f"- {class_name}")
                
        if empty:
            print("\nPotentially empty or incomplete files:")
            for file_path in empty:
                print(f"- {file_path}")
    else:
        print("All classes appear to be properly distributed!")