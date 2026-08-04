"""
project_structure.py - Detect and display project structure
Run this to see your complete project folder layout
"""

import os
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIG — Set your project root path
# ══════════════════════════════════════════════════════════════

# Auto-detect or manually set
PROJECT_ROOT = r"E:\SEM 6\Final Year Project 1\FYP\FYP\smartDyslexiaDetector_v2"

# If you want to use current directory instead, uncomment:
# PROJECT_ROOT = os.getcwd()

# ══════════════════════════════════════════════════════════════


def get_folder_structure(path, indent=0, max_depth=3, current_depth=0, exclude=None):
    """
    Recursively get folder structure as a string.
    """
    if exclude is None:
        exclude = ['.git', '__pycache__', 'venv', 'env', '.venv', '.idea', '.vscode', 'node_modules']
    
    if current_depth > max_depth:
        return ""
    
    result = ""
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return result
    
    for item in items:
        if item in exclude:
            continue
        
        full_path = os.path.join(path, item)
        prefix = "│   " * indent
        is_dir = os.path.isdir(full_path)
        
        if is_dir:
            result += f"{prefix}├── 📁 {item}/\n"
            result += get_folder_structure(full_path, indent + 1, max_depth, current_depth + 1, exclude)
        else:
            # Get file size
            try:
                size = os.path.getsize(full_path)
                size_str = f" ({size:,} bytes)" if size > 0 else ""
            except:
                size_str = ""
            
            # Show extension for important files
            ext = os.path.splitext(item)[1].lower()
            important_exts = ['.py', '.yaml', '.yml', '.txt', '.md', '.json', '.html', '.css', '.js', '.pt', '.pth', '.sql']
            if ext in important_exts:
                result += f"{prefix}├── 📄 {item}{size_str}\n"
            else:
                result += f"{prefix}├── {item}{size_str}\n"
    
    return result


def check_dataset_structure(base_path):
    """Check and report dataset structure."""
    print("\n" + "=" * 70)
    print("DATASET STRUCTURE CHECK")
    print("=" * 70)
    
    # Common dataset paths to check
    dataset_paths = [
        os.path.join(base_path, "synthetic_dyslexia_dataset"),
        os.path.join(base_path, "dataset"),
        os.path.join(base_path, "data"),
    ]
    
    found = False
    for dpath in dataset_paths:
        if os.path.exists(dpath):
            print(f"\n✅ Found dataset at: {dpath}")
            found = True
            
            # Check for data.yaml
            yaml_path = os.path.join(dpath, "data.yaml")
            if os.path.exists(yaml_path):
                print(f"   ✅ data.yaml found at: {yaml_path}")
                # Read and display content
                try:
                    with open(yaml_path, 'r') as f:
                        content = f.read()
                        print(f"\n   Content of data.yaml:")
                        print("   " + "-" * 50)
                        for line in content.split('\n')[:20]:
                            print(f"   {line}")
                        print("   " + "-" * 50)
                except:
                    pass
            else:
                print(f"   ❌ data.yaml NOT found at: {yaml_path}")
            
            # Check images and labels
            for subfolder in ['images', 'labels']:
                sub_path = os.path.join(dpath, subfolder)
                if os.path.exists(sub_path):
                    print(f"\n   📁 {subfolder}/")
                    for split in ['train', 'val', 'test']:
                        split_path = os.path.join(sub_path, split)
                        if os.path.exists(split_path):
                            count = len([f for f in os.listdir(split_path) if os.path.isfile(os.path.join(split_path, f))])
                            print(f"      ✅ {split}: {count} files")
                        else:
                            print(f"      ❌ {split}: NOT FOUND")
                else:
                    print(f"   ❌ {subfolder}/ NOT FOUND")
            
            # Check if there's a nested dataset folder
            nested = os.path.join(dpath, "synthetic_dyslexia_dataset")
            if os.path.exists(nested):
                print(f"\n   ⚠️ Nested folder found: {nested}")
                print(f"      This might be causing path issues!")
    
    if not found:
        print("\n❌ No dataset folder found!")
        print("   Common locations checked:")
        for dpath in dataset_paths:
            print(f"   - {dpath}")


def list_py_files(base_path):
    """List all Python files in the project."""
    print("\n" + "=" * 70)
    print("PYTHON FILES")
    print("=" * 70)
    
    py_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip virtual environments and caches
        skip_dirs = ['venv', 'env', '__pycache__', '.git', 'node_modules']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                py_files.append(rel_path)
    
    if py_files:
        for f in sorted(py_files):
            print(f"   📄 {f}")
    else:
        print("   No Python files found.")


def check_model_files(base_path):
    """Check for model files."""
    print("\n" + "=" * 70)
    print("MODEL FILES")
    print("=" * 70)
    
    model_extensions = ['.pt', '.pth', '.onnx', '.pb', '.h5', '.keras', '.pkl', '.joblib']
    model_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in model_extensions:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                size = os.path.getsize(full_path) / (1024 * 1024)
                model_files.append(f"{rel_path} ({size:.2f} MB)")
    
    if model_files:
        for f in sorted(model_files):
            print(f"   ✅ {f}")
    else:
        print("   No model files found.")


def check_database_files(base_path):
    """Check for database files."""
    print("\n" + "=" * 70)
    print("DATABASE FILES")
    print("=" * 70)
    
    db_files = []
    db_extensions = ['.sql', '.db', '.sqlite', '.sqlite3']
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in db_extensions:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                db_files.append(rel_path)
    
    if db_files:
        for f in sorted(db_files):
            print(f"   📄 {f}")
    else:
        print("   No database files found.")


def main():
    print("=" * 70)
    print("PROJECT STRUCTURE DETECTOR")
    print("=" * 70)
    
    # Check if project root exists
    if not os.path.exists(PROJECT_ROOT):
        print(f"\n❌ Project root not found: {PROJECT_ROOT}")
        print("\nPlease update PROJECT_ROOT variable in the script.")
        sys.exit(1)
    
    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    
    # 1. Full folder structure
    print("\n" + "=" * 70)
    print("FULL FOLDER STRUCTURE (Limited to 4 levels)")
    print("=" * 70)
    print()
    print(PROJECT_ROOT + "/")
    structure = get_folder_structure(PROJECT_ROOT, indent=0, max_depth=4)
    print(structure)
    
    # 2. Dataset check
    check_dataset_structure(PROJECT_ROOT)
    
    # 3. Python files
    list_py_files(PROJECT_ROOT)
    
    # 4. Model files
    check_model_files(PROJECT_ROOT)
    
    # 5. Database files
    check_database_files(PROJECT_ROOT)
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Python Version: {sys.version}")
    print(f"Current Directory: {os.getcwd()}")
    
    # Check if we're running from the project root
    if os.getcwd() != PROJECT_ROOT:
        print(f"\n⚠️ You are running this script from:")
        print(f"   {os.getcwd()}")
        print(f"\n   The script might not work correctly if paths are relative.")
        print(f"   Consider running from: {PROJECT_ROOT}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()