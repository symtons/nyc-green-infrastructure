"""Step 7 entry point — run the trained U-Net across the full NYC raster."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_green.config import load_config
from nyc_green.inference import (
    load_trained_model,
    prepare_input_raster,
    sliding_window_inference,
    probs_to_classes,
    apply_boundary_mask,
    write_landcover_raster,
    compare_to_worldcover,
)


def main():
    print("=" * 60)
    print("STEP 7 — MODEL INFERENCE ON FULL NYC RASTER")
    print("=" * 60)

    cfg = load_config()
    mcfg = cfg["model"]

    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    outputs_dir = cfg["paths"]["outputs_dir"]

    landsat_path = interim_dir / "nyc_landsat_30m.tif"
    worldcover_path = interim_dir / "nyc_worldcover_30m.tif"
    checkpoint_path = models_dir / "best.pt"

    for p in (landsat_path, worldcover_path, checkpoint_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Load model ---
    print("\n[1/4] Loading trained model...")
    model = load_trained_model(
        checkpoint_path,
        num_classes=mcfg["num_classes"],
        in_channels=mcfg["in_channels"],
        device=device,
    )
    print(f"  ✓ Loaded from {checkpoint_path.name}")

    # --- Load and prepare input raster ---
    print("\n[2/4] Preparing input raster...")
    image, profile = prepare_input_raster(landsat_path)
    print(f"  Shape: {image.shape}  (channels, H, W)")

    # --- Run sliding-window inference ---
    print("\n[3/4] Running sliding-window inference...")
    prob_map = sliding_window_inference(
        model,
        image,
        tile_size=mcfg["tile_size"],
        stride=mcfg["tile_size"] // 2,   # 50% overlap is plenty for inference smoothing
        num_classes=mcfg["num_classes"],
        batch_size=mcfg["batch_size"],
        device=device,
    )
    class_map = probs_to_classes(prob_map)

    # Mask outside NYC boundary
    class_map = apply_boundary_mask(class_map, landsat_path, nodata_value=255)

    out_path = processed_dir / "nyc_landcover_predicted.tif"
    write_landcover_raster(class_map, profile, out_path)
    print(f"  ✓ Saved: {out_path.name}  shape={class_map.shape}")

    # Class distribution in the prediction
    valid = class_map != 255
    total = int(valid.sum())
    class_names = ["Vegetation", "Water", "Built-up"]
    print("\n  Predicted class distribution (valid NYC pixels):")
    for c, name in enumerate(class_names):
        n = int(((class_map == c) & valid).sum())
        pct = 100.0 * n / max(total, 1)
        print(f"    {name:12s} {n:>9,} pixels ({pct:5.2f}%)")

    # --- Compare to WorldCover ---
    print("\n[4/4] Comparing to ESA WorldCover baseline...")
    comp = compare_to_worldcover(
        out_path,
        worldcover_path,
        num_classes=mcfg["num_classes"],
    )
    print(f"  Total valid pixels compared: {comp['total_pixels']:,}")
    print(f"  Pixelwise agreement: {comp['agreement']*100:.2f}%")
    print(f"  Per-class IoU (vs WorldCover):")
    for name, iou in zip(class_names, comp["iou_per_class"]):
        print(f"    {name:12s} {iou:.4f}")
    print(f"  Confusion matrix (rows=WorldCover truth, cols=model prediction):")
    cm = np.array(comp["confusion_matrix"])
    header = "            " + "  ".join(f"{n:>10s}" for n in class_names)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10,}" for v in row)
        print(f"  {class_names[i]:12s} {row_str}")

    # Save comparison JSON
    comp_out = outputs_dir / "analysis"
    comp_out.mkdir(parents=True, exist_ok=True)
    with open(comp_out / "model_vs_worldcover.json", "w") as f:
        json.dump(comp, f, indent=2)
    print(f"\n  ✓ Comparison saved: {comp_out / 'model_vs_worldcover.json'}")

    print("\n" + "=" * 60)
    print("✓ STEP 7 COMPLETE")
    print("=" * 60)
    print(f"  Predicted land cover: {out_path}")
    print(f"  Comparison metrics:   {comp_out / 'model_vs_worldcover.json'}")


if __name__ == "__main__":
    main()