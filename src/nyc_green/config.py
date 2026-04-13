"""Config loading utilities."""
from pathlib import Path
import yaml

# Project root = two levels up from this file
# (src/nyc_green/config.py -> src/nyc_green -> src -> PROJECT_ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config and resolve relative paths to absolute paths.

    Returns a dict with the same shape as config.yaml, but every path
    under paths.raw.* and paths.*_dir is converted to an absolute Path
    rooted at the project root.
    """
    config_path = PROJECT_ROOT / path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve directory paths
    for key in ("raw_dir", "interim_dir", "processed_dir", "outputs_dir", "models_dir"):
        config["paths"][key] = PROJECT_ROOT / config["paths"][key]

    # Resolve individual raw file paths
    for key, rel_path in config["paths"]["raw"].items():
        config["paths"]["raw"][key] = PROJECT_ROOT / rel_path

    return config


if __name__ == "__main__":
    cfg = load_config()
    print("=" * 60)
    print("CONFIG LOADED")
    print("=" * 60)
    print(f"Project:     {cfg['project']['name']}")
    print(f"Study area:  {cfg['project']['study_area']}")
    print(f"CRS:         {cfg['project']['crs']}")
    print(f"Resolution:  {cfg['project']['target_resolution_m']}m")
    print()
    print("Priority weights:")
    total = 0.0
    for k, v in cfg["priority_weights"].items():
        print(f"  {k:20s} {v}")
        total += v
    print(f"  {'TOTAL':20s} {total}")
    assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"
    print()
    print("Raw data files:")
    for k, v in cfg["paths"]["raw"].items():
        mark = "✓" if v.exists() else "✗ MISSING"
        print(f"  {mark}  {k:25s} {v.name}")
    print()
    print("Model config:")
    print(f"  architecture: {cfg['model']['architecture']} ({cfg['model']['encoder']} + {cfg['model']['encoder_weights']})")
    print(f"  in_channels:  {cfg['model']['in_channels']}")
    print(f"  tile_size:    {cfg['model']['tile_size']}")
    print(f"  batch_size:   {cfg['model']['batch_size']}")