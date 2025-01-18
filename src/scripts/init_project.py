from pathlib import Path

def init_project():
    """Initialize project directory structure and packages"""
    # Create package directories
    packages = [
        "src",
        "src/api",
        "src/ingestion",
        "src/vectorstore",
        "src/rag",
        "src/models"
    ]
    
    # Create __init__.py files
    for package in packages:
        Path(package).mkdir(parents=True, exist_ok=True)
        init_file = Path(f"{package}/__init__.py")
        if not init_file.exists():
            init_file.touch()

if __name__ == "__main__":
    init_project() 