import shutil
from pathlib import Path

def distribute_code():
    base_dir = Path("e:/Desktop/docker/dev/cerebro/zerolm")
    
    # First, ensure we have a backup
    if not (base_dir / "zerolm.py.backup").exists():
        shutil.copy(base_dir / "zerolm.py", base_dir / "zerolm.py.backup")

    # Create the distribution mapping
    distributions = {
        "core/types.py": [
            "ResponseType",
            "Response",
            "ValidationResult",
            "MemoryStats"
        ],
        
        "memory/cache.py": [
            "ConcurrentLRUCache"
        ],
        
        "memory/manager.py": [
            "AdaptiveMemoryManager"
        ],
        
        "vector/manager.py": [
            "VectorManager"
        ],
        
        "vector/similarity.py": [
            "PatternMatcher"
        ],
        
        "context/tracker.py": [
            "HierarchicalContext"
        ],
        
        "context/weighting.py": [
            "TemporalWeightingSystem",
            "ContextWeighter"
        ],
        
        "learning/validator.py": [
            "LearningValidator"
        ],
        
        "learning/patterns.py": [
            "AutoCorrector",
            "TemplateEnforcer",
            "TemplateValidator"
        ],
        
        "core/model.py": [
            "ZeroShotLM"  # Main class goes last as it depends on others
        ]
    }

    # Read the original file
    with open(base_dir / "zerolm.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Helper function to extract class and its imports
    def extract_class(class_name, content):
        import_lines = []
        class_lines = []
        in_class = False
        
        for line in content.split("\n"):
            if line.startswith("import ") or line.startswith("from "):
                import_lines.append(line)
            elif line.startswith(f"class {class_name}"):
                in_class = True
                class_lines.append(line)
            elif in_class:
                if line.startswith("class "):
                    break
                class_lines.append(line)
        
        return "\n".join(import_lines), "\n".join(class_lines)

    # Distribute the code
    for file_path, classes in distributions.items():
        full_path = base_dir / "zerolm" / file_path
        
        # Create directory if it doesn't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine all classes for this file
        file_content = ['"""Module containing {}."""'.format(", ".join(classes))]
        imports = set()
        class_contents = []
        
        for class_name in classes:
            imp, cls = extract_class(class_name, content)
            imports.update(imp.split("\n"))
            class_contents.append(cls)
        
        # Write the file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(file_content))
            f.write("\n\n")
            f.write("\n".join(sorted(filter(None, imports))))
            f.write("\n\n")
            f.write("\n\n".join(class_contents))

if __name__ == "__main__":
    distribute_code()