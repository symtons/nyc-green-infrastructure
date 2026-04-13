"""Config loading utilities."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = "config/config.yaml") -> dict:
    """Load the YAML config and resolve all paths to absolute paths."""
    config_path = PROJECT_ROOT / path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve raw paths to absolute paths rooted at project root
    for key, rel_path in config["paths"]["raw"].items():
        config["paths"]["raw"][key] = PROJECT_ROOT / rel_path

    for key in ("interim", "processed", "outputs"):
        config["paths"][key] = PROJECT_ROOT / config["paths"][key]

    return config


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully")
    print(f"  Project: {cfg['project']['name']}")
    print(f"  Study area: {cfg['project']['study_area']}")
    print(f"  Priority weights: {cfg['priority_weights']}")
    print(f"  Raw paths resolved:")
    for k, v in cfg["paths"]["raw"].items():
        exists = "✓" if v.exists() else "✗ MISSING"
        print(f"    {exists} {k}: {v}")