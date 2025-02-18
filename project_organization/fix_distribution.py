import shutil
from pathlib import Path
from verify_distribution import verify_distribution

def fix_distribution():
    base_dir = Path("e:/Desktop/docker/dev/cerebro/zerolm")
    
    # First, backup the current structure
    backup_dir = base_dir / "backup_modules"
    backup_dir.mkdir(exist_ok=True)
    
    for item in (base_dir / "zerolm").glob("**/*"):
        if item.is_file() and item.suffix == ".py":
            rel_path = item.relative_to(base_dir / "zerolm")
            backup_path = backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, backup_path)
    
    # Re-run the distribution script with complete class mapping
    from distribute_modules import distribute_code
    distribute_code()
    
    # Verify the fix
    missing, empty = verify_distribution()
    return not (missing or empty)

if __name__ == "__main__":
    success = fix_distribution()
    if success:
        print("Distribution has been fixed successfully!")
    else:
        print("Some issues remain. Please check the verification output.")