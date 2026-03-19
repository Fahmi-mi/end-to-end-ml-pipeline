from pathlib import Path


def setup_data_directories(project_root: Path) -> None:
    """Create required data directories and remove .gitkeep files."""
    data_dirs = [
        project_root / "data",
        project_root / "data" / "raw",
        project_root / "data" / "processed",
    ]

    created_dirs = []
    existing_dirs = []

    for directory in data_dirs:
        if directory.exists():
            existing_dirs.append(directory)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)

    removed_gitkeeps = []
    for gitkeep_file in project_root.rglob(".gitkeep"):
        if gitkeep_file.is_file():
            gitkeep_file.unlink()
            removed_gitkeeps.append(gitkeep_file)

    print("=== Data Directory Setup ===")

    if created_dirs:
        print("Created directories:")
        for directory in created_dirs:
            print(f"  - {directory.relative_to(project_root)}")
    else:
        print("No new directories created (all already exist).")

    if existing_dirs:
        print("Already existed:")
        for directory in existing_dirs:
            print(f"  - {directory.relative_to(project_root)}")

    if removed_gitkeeps:
        print("Removed .gitkeep files:")
        for file_path in removed_gitkeeps:
            print(f"  - {file_path.relative_to(project_root)}")
    else:
        print("No .gitkeep files found.")

    print("Setup complete.")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    setup_data_directories(project_root)


if __name__ == "__main__":
    main()
