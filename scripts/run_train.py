"""Step 6 entry point — train the U-Net on tiles produced by Step 5."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_green.config import load_config
from nyc_green.dataset import TileDataset, make_balanced_sampler
from nyc_green.losses import FocalLoss
from nyc_green.metrics import SegmentationMetrics
from nyc_green.model import build_unet, count_parameters


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, loader, loss_fn, metrics, device):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  val  ", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()
            n_batches += 1
            metrics.update(logits, masks)
    return total_loss / max(n_batches, 1), metrics.compute()


def main():
    print("=" * 60)
    print("STEP 6 — MODEL TRAINING")
    print("=" * 60)

    cfg = load_config()
    mcfg = cfg["model"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    tiles_dir = cfg["paths"]["processed_dir"] / "tiles"
    models_dir = cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    print("\nBuilding datasets...")
    train_ds = TileDataset(tiles_dir, split="train", augment=True)
    val_ds   = TileDataset(tiles_dir, split="val",   augment=False)
    test_ds  = TileDataset(tiles_dir, split="test",  augment=False)
    print(f"  train: {len(train_ds)} tiles (augmented)")
    print(f"  val:   {len(val_ds)} tiles")
    print(f"  test:  {len(test_ds)} tiles")

    # Balanced sampler for training (oversamples tiles with water)
    if mcfg.get("use_balanced_sampler", True):
        sampler = make_balanced_sampler(train_ds, rare_class_col="water_frac", boost=5.0)
        train_loader = DataLoader(
            train_ds,
            batch_size=mcfg["batch_size"],
            sampler=sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        print("  Using weighted sampler (boost=5.0 on water_frac)")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=mcfg["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    val_loader  = DataLoader(val_ds,  batch_size=mcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=mcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Model
    print("\nBuilding model...")
    model = build_unet(
        num_classes=mcfg["num_classes"],
        in_channels=mcfg["in_channels"],
        encoder_name=mcfg["encoder"],
        encoder_weights=mcfg["encoder_weights"],
    ).to(device)
    print(f"  Architecture: {mcfg['architecture']} + {mcfg['encoder']} ({mcfg['encoder_weights']})")
    print(f"  Parameters:   {count_parameters(model):,}")

    # Loss
    alpha = torch.tensor(mcfg["focal_loss_alpha"], dtype=torch.float32, device=device)
    loss_fn = FocalLoss(alpha=alpha, gamma=mcfg["focal_loss_gamma"], ignore_index=255).to(device)
    print(f"  Loss: FocalLoss(alpha={mcfg['focal_loss_alpha']}, gamma={mcfg['focal_loss_gamma']})")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Metrics
    metrics = SegmentationMetrics(num_classes=mcfg["num_classes"], ignore_index=255)

    # Training loop
    print("\n" + "=" * 60)
    print(f"TRAINING — up to {mcfg['max_epochs']} epochs, early stop patience {mcfg['early_stopping_patience']}")
    print("=" * 60)

    best_val_loss = float("inf")
    best_val_metrics = None
    epochs_since_improvement = 0
    history = []
    t_start = time.time()

    for epoch in range(1, mcfg["max_epochs"] + 1):
        t_epoch = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, metrics, device)
        scheduler.step(val_loss)
        dt = time.time() - t_epoch

        iou_str = " / ".join(f"{x:.3f}" for x in val_metrics["iou_per_class"])
        print(
            f"Epoch {epoch:2d}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mIoU={val_metrics['mean_iou']:.3f}  "
            f"IoU[veg/water/built]={iou_str}  "
            f"({dt:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_iou": val_metrics["mean_iou"],
            "val_iou_per_class": val_metrics["iou_per_class"],
            "val_accuracy": val_metrics["accuracy"],
        })

        # Checkpoint if this is the best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            epochs_since_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "config": mcfg,
            }, models_dir / "best.pt")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= mcfg["early_stopping_patience"]:
                print(f"\nEarly stopping — no improvement in {mcfg['early_stopping_patience']} epochs")
                break

    t_total = time.time() - t_start
    print(f"\nTraining finished in {t_total:.1f}s ({t_total/60:.1f} min)")

    # Load best checkpoint and evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (loaded from best checkpoint)")
    print("=" * 60)
    ckpt = torch.load(models_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, metrics, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test mean IoU: {test_metrics['mean_iou']:.4f}")
    print("Test IoU per class:")
    class_names = ["Vegetation", "Water     ", "Built-up  "]
    for name, iou in zip(class_names, test_metrics["iou_per_class"]):
        print(f"  {name}: {iou:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = np.array(test_metrics["confusion_matrix"])
    for i, row in enumerate(cm):
        print(f"  {class_names[i]}: {row.tolist()}")

    # Save history and final metrics
    with open(models_dir / "training_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "total_time_sec": t_total,
        }, f, indent=2)

    print(f"\n✓ Best checkpoint: {models_dir / 'best.pt'}")
    print(f"✓ Training history: {models_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()