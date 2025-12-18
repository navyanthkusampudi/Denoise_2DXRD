from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from mask import Masker
from models.babyunet import BabyUnet
from models.singleconv import SingleConvolution
from models.unet import Unet
from models.dncnn import DnCNN
from pipeline.dataloaders import make_loader
from pipeline.h5_dataset import H5StackDataset


def _make_grad_scaler(*, device: torch.device, enabled: bool):
    """
    Create a GradScaler in a way that's compatible across PyTorch versions.

    - Newer PyTorch supports: torch.amp.GradScaler(device_type=..., enabled=...)
    - Older PyTorch may only support: torch.cuda.amp.GradScaler(enabled=...)
    """
    if not enabled:
        # Any GradScaler works when disabled; pick whichever API exists.
        try:
            # Newer torch accepts a device type as first positional arg.
            return torch.amp.GradScaler(device.type, enabled=False)  # type: ignore[attr-defined]
        except Exception:
            try:
                return torch.amp.GradScaler(enabled=False)  # type: ignore[attr-defined]
            except Exception:
                return torch.cuda.amp.GradScaler(enabled=False)

    # enabled=True is only used for CUDA in this script (see use_amp).
    try:
        # Preferred API: torch.amp.GradScaler('cuda', enabled=True)
        return torch.amp.GradScaler(device.type, enabled=True)  # type: ignore[attr-defined]
    except TypeError:
        # Some versions require device_type= kwarg.
        try:
            return torch.amp.GradScaler(device_type=device.type, enabled=True)  # type: ignore[attr-defined]
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=True)
    except AttributeError:
        # torch.amp may not exist on very old versions
        return torch.cuda.amp.GradScaler(enabled=True)


@dataclass(frozen=True)
class TrainConfig:
    h5_path: str
    max_samples: Optional[int]
    model: str
    in_channels: int
    out_channels: int
    patch: Optional[Tuple[int, int]]
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    masker_width: int
    masker_mode: str
    include_mask_as_input: bool
    device: str
    amp: bool
    num_workers: int
    seed: int
    save_dir: str
    log_every: int
    val_frac: float
    val_max_batches: int
    early_stop_patience: int
    early_stop_min_delta: float
    save_best: bool
    # Data scaling/normalization (kept as defaults for backward compatibility with older ckpts).
    # - scale_integers=True: integer HDF5 dtypes (e.g. uint16) are scaled to [0,1] by /dtype_max.
    # - normalize='percentile': per-sample robust scaling using p1..p99 (NOT invertible to counts).
    # For reconstructable "denoised counts", prefer: scale_integers=False and normalize='none'.
    normalize: str = "percentile"
    scale_integers: bool = True
    # Backward-compatible defaults so old checkpoints (saved config dicts) can be reloaded.
    stdout_epoch_only: bool = False
    force_tqdm: bool = False


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    name = cfg.model.lower()

    # Keep this simple and explicit (avoids accidental factory mismatch).
    if name == "unet":
        return Unet(cfg.in_channels, cfg.out_channels)
    if name in {"babyunet", "baby-unet"}:
        return BabyUnet(cfg.in_channels, cfg.out_channels)
    if name in {"conv", "convolution", "singleconv"}:
        # Width is the kernel size; kept modest by default.
        return SingleConvolution(cfg.in_channels, cfg.out_channels, width=3)
    if name in {"dncnn"}:
        return DnCNN(cfg.in_channels, cfg.out_channels)

    raise ValueError(f"Unknown model: {cfg.model!r} (use: unet, baby-unet, convolution, dncnn)")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Noise2Self training on a detector-image stack stored in HDF5.")
    p.add_argument(
        "--h5",
        required=True,
        help="Path to .h5 file containing dataset '/data' (preferred) or '/X' with shape (N,H,W) or (N0,N1,H,W).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, train using only the first N frames from the stack.",
    )

    p.add_argument("--model", default="unet", help="Model: unet | baby-unet | convolution")
    p.add_argument("--in-ch", type=int, default=1, help="Input channels (usually 1).")
    p.add_argument("--out-ch", type=int, default=1, help="Output channels (usually 1).")

    p.add_argument("--patch", type=int, nargs="*", default=[128], help="Patch size: P or H W. Use 0 for full frame.")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument(
        "--normalize",
        default="percentile",
        choices=["none", "minmax", "percentile"],
        help=(
            "Per-sample normalization applied after optional integer scaling. "
            "'percentile' uses p1..p99 and is NOT uniquely invertible to counts; "
            "for reconstructable denoised counts, use --normalize none."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--scale-integers",
        dest="scale_integers",
        action="store_true",
        help="Scale integer inputs (e.g. uint16) to [0,1] by dividing by dtype max (default).",
    )
    g.add_argument(
        "--no-scale-integers",
        dest="scale_integers",
        action="store_false",
        help="Do NOT scale integer inputs; train directly in counts units (recommended for reconstructable outputs).",
    )
    p.set_defaults(scale_integers=True)

    p.add_argument("--masker-width", type=int, default=4, help="Mask grid width (width^2 masks).")
    p.add_argument("--masker-mode", default="interpolate", choices=["zero", "interpolate"])
    p.add_argument("--include-mask", action="store_true", help="Concatenate the mask as an extra input channel.")

    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save-dir", default="runs/noise2self")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="Fraction of frames to hold out for validation (0 disables validation).",
    )
    p.add_argument(
        "--val-max-batches",
        type=int,
        default=0,
        help="If >0, limit validation to the first K batches each epoch (faster, noisier).",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if val loss doesn't improve for this many epochs (0 disables early stopping).",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum decrease in val loss to be considered an improvement.",
    )
    p.add_argument(
        "--save-best",
        action="store_true",
        help="Also write a ckpt_best.pt whenever validation improves (requires --val-frac > 0).",
    )
    p.add_argument(
        "--stdout-epoch-only",
        action="store_true",
        help="Reduce stdout to only epoch_end/saved/saved_best/early_stop lines (keeps Slurm .out small).",
    )
    p.add_argument(
        "--tqdm",
        dest="force_tqdm",
        action="store_true",
        help="Force tqdm progress bar to stderr (even when stderr is not a TTY).",
    )

    a = p.parse_args()

    if a.max_samples is not None and a.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer")
    if not (0.0 <= float(a.val_frac) < 1.0):
        raise ValueError("--val-frac must be in [0, 1). Use 0 to disable validation.")
    if a.val_max_batches is not None and int(a.val_max_batches) < 0:
        raise ValueError("--val-max-batches must be >= 0")
    if a.early_stop_patience is not None and int(a.early_stop_patience) < 0:
        raise ValueError("--early-stop-patience must be >= 0")

    patch_hw: Optional[Tuple[int, int]]
    if len(a.patch) == 1:
        patch_hw = None if a.patch[0] == 0 else (int(a.patch[0]), int(a.patch[0]))
    elif len(a.patch) == 2:
        patch_hw = (int(a.patch[0]), int(a.patch[1]))
    else:
        raise ValueError("--patch expects: P  OR  H W  OR  0")

    return TrainConfig(
        h5_path=str(a.h5),
        max_samples=a.max_samples,
        model=a.model,
        in_channels=a.in_ch,
        out_channels=a.out_ch,
        patch=patch_hw,
        batch_size=a.batch,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        normalize=str(a.normalize),
        scale_integers=bool(a.scale_integers),
        masker_width=a.masker_width,
        masker_mode=a.masker_mode,
        include_mask_as_input=bool(a.include_mask),
        device=a.device,
        amp=bool(a.amp),
        num_workers=a.workers,
        seed=a.seed,
        save_dir=a.save_dir,
        log_every=a.log_every,
        val_frac=float(a.val_frac),
        val_max_batches=int(a.val_max_batches),
        early_stop_patience=int(a.early_stop_patience),
        early_stop_min_delta=float(a.early_stop_min_delta),
        save_best=bool(a.save_best),
        stdout_epoch_only=bool(a.stdout_epoch_only),
        force_tqdm=bool(a.force_tqdm),
    )


def _split_indices(frame_indices: np.ndarray, *, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split frame indices into (train, val) deterministically."""
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if val_frac <= 0.0 or frame_indices.size == 0:
        return frame_indices, np.asarray([], dtype=np.int64)
    n = int(frame_indices.size)
    n_val = int(round(n * float(val_frac)))
    n_val = max(1, min(n - 1, n_val))  # ensure both splits non-empty
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return frame_indices[train_idx], frame_indices[val_idx]


@torch.no_grad()
def _eval_epoch_loss(
    *,
    model: torch.nn.Module,
    dl,
    masker: Masker,
    device: torch.device,
    use_amp: bool,
    max_batches: int,
) -> float:
    """Compute mean masked MSE over a loader."""
    model.eval()
    losses: list[float] = []
    for bidx, batch in enumerate(dl):
        if max_batches > 0 and bidx >= max_batches:
            break
        x, _meta = batch
        x = x.to(device, non_blocking=True)
        # Deterministic mask index to reduce eval noise across epochs.
        i = int(bidx % len(masker))
        net_in, mask = masker.mask(x, i)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(net_in)
            loss = F.mse_loss(pred * mask, x * mask)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    cfg = parse_args()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = bool(cfg.amp and device.type == "cuda")

    def _count_params(m: torch.nn.Module) -> tuple[int, int]:
        total = int(sum(p.numel() for p in m.parameters()))
        trainable = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
        return total, trainable

    def _render_bar(done: int, total: int, width: int = 30) -> str:
        if total <= 0:
            return "[" + ("?" * width) + "]"
        frac = max(0.0, min(1.0, float(done) / float(total)))
        filled = int(round(frac * width))
        if filled <= 0:
            return "[" + (">" + "." * (width - 1)) + "]"
        if filled >= width:
            return "[" + ("=" * width) + "]"
        return "[" + ("=" * (filled - 1) + ">" + "." * (width - filled)) + "]"

    # Select dataset backend + optionally restrict to first N frames.
    ds_full = H5StackDataset(
        cfg.h5_path,
        patch_size=cfg.patch,
        normalize=cfg.normalize,
        scale_integers=cfg.scale_integers,
        augment=True,
        seed=cfg.seed,
    )
    if cfg.max_samples is None:
        ds = ds_full
    else:
        n = min(int(cfg.max_samples), len(ds_full))
        ds = H5StackDataset(
            cfg.h5_path,
            patch_size=cfg.patch,
            normalize=cfg.normalize,
            scale_integers=cfg.scale_integers,
            augment=True,
            seed=cfg.seed,
            indices=list(range(n)),
        )

    # If include-mask, the model input channels must account for the extra channel.
    in_ch = cfg.in_channels + (1 if cfg.include_mask_as_input else 0)
    cfg2 = TrainConfig(**{**cfg.__dict__, "in_channels": in_ch})

    # Split into train/val frames (optional). We split by underlying frame indices to avoid leakage.
    frame_indices = getattr(ds, "indices", None)
    if frame_indices is None:
        frame_indices = np.arange(len(ds), dtype=np.int64)
    train_frames, val_frames = _split_indices(np.asarray(frame_indices), val_frac=cfg2.val_frac, seed=cfg2.seed)

    if not cfg2.stdout_epoch_only:
        if cfg2.val_frac > 0.0:
            print(f"split: train_frames={len(train_frames)} val_frames={len(val_frames)} (val_frac={cfg2.val_frac})")
        else:
            print("split: validation disabled (--val-frac 0)")

    train_ds = H5StackDataset(
        cfg2.h5_path,
        patch_size=cfg2.patch,
        normalize=cfg2.normalize,
        scale_integers=cfg2.scale_integers,
        augment=True,
        seed=cfg2.seed,
        indices=train_frames.tolist(),
    )
    val_ds = (
        None
        if val_frames.size == 0
        else H5StackDataset(
            cfg2.h5_path,
            patch_size=cfg2.patch,
            normalize=cfg2.normalize,
            scale_integers=cfg2.scale_integers,
            augment=False,
            seed=cfg2.seed,
            indices=val_frames.tolist(),
        )
    )

    train_dl = make_loader(
        train_ds,
        batch_size=cfg2.batch_size,
        shuffle=True,
        num_workers=cfg2.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_dl = (
        None
        if val_ds is None
        else make_loader(
            val_ds,
            batch_size=cfg2.batch_size,
            shuffle=False,
            num_workers=cfg2.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
    )

    model = build_model(cfg2).to(device)
    total_params, trainable_params = _count_params(model)
    if not cfg2.stdout_epoch_only:
        print("==== Model Summary ====")
        print(f"model_name: {cfg2.model}")
        print(f"in_channels: {cfg2.in_channels} out_channels: {cfg2.out_channels}")
        print(f"total_params: {total_params}")
        print(f"trainable_params: {trainable_params}")
        print("architecture:")
        print(model)
        print("=======================")

    opt = torch.optim.Adam(model.parameters(), lr=cfg2.lr, weight_decay=cfg2.weight_decay)
    scaler = _make_grad_scaler(device=device, enabled=use_amp)

    masker = Masker(
        width=cfg2.masker_width,
        mode=cfg2.masker_mode,
        infer_single_pass=True,
        include_mask_as_input=cfg2.include_mask_as_input,
    )

    os.makedirs(cfg2.save_dir, exist_ok=True)

    step = 0
    model.train()
    best_val: float = float("inf")
    best_epoch: int = 0
    epochs_since_improve: int = 0
    for epoch in range(cfg2.epochs):
        epoch_t0 = time.perf_counter()
        # Track mean training loss per epoch (more useful than a single step print).
        train_losses: list[float] = []
        n_batches = len(train_dl)

        # tqdm is great interactively; if --tqdm is set we use it even under Slurm.
        use_tqdm = (tqdm is not None) and (cfg2.force_tqdm or sys.stderr.isatty())
        if use_tqdm:
            dl_iter = tqdm(
                enumerate(train_dl),
                total=n_batches,
                desc=f"epoch {epoch+1}/{cfg2.epochs}",
                file=sys.stderr,
                leave=False,
                # Under Slurm stderr is captured to a file: throttle to keep .err size reasonable.
                mininterval=5.0,
            )
        else:
            dl_iter = enumerate(train_dl)
            # Print ~20 lines per epoch to stderr (keeps .err small).
            print_every = max(1, n_batches // 20) if n_batches > 0 else 1

        for bidx, batch in dl_iter:
            x, _meta = batch
            x = x.to(device, non_blocking=True)

            # Choose a random mask pattern per step (simple + works well).
            i = int(torch.randint(low=0, high=len(masker), size=(1,)).item())
            net_in, mask = masker.mask(x, i)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(net_in)
                loss = F.mse_loss(pred * mask, x * mask)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Progress reporting:
            # - Interactive: tqdm progress bar to stderr
            # - Slurm logs: sparse stderr updates to avoid massive .err files
            if use_tqdm:
                # Update occasionally; tqdm handles display.
                if (bidx + 1) % max(1, cfg2.log_every) == 0 or (bidx + 1) == n_batches:
                    try:
                        dl_iter.set_postfix(loss=f"{loss.item():.6f}", refresh=False)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            else:
                if n_batches > 0 and ((bidx + 1) % print_every == 0 or (bidx + 1) == 1 or (bidx + 1) == n_batches):
                    elapsed = time.perf_counter() - epoch_t0
                    print(
                        f"epoch {epoch+1}/{cfg2.epochs} batch {bidx+1}/{n_batches} "
                        f"loss={loss.item():.6f} elapsed={elapsed:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )

            if (not cfg2.stdout_epoch_only) and (step % cfg2.log_every == 0):
                print(
                    f"epoch={epoch+1}/{cfg2.epochs} step={step} "
                    f"loss={loss.item():.6f} device={device.type} amp={use_amp}"
                )
            train_losses.append(float(loss.item()))
            step += 1

        train_epoch_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        epoch_elapsed = time.perf_counter() - epoch_t0
        # Epoch summary: optional (many users want only epoch_end/saved lines in .out).
        if not cfg2.stdout_epoch_only:
            print(f"epoch {epoch+1}/{cfg2.epochs} done  time={epoch_elapsed:.2f}s  train_loss={train_epoch_loss:.6f}")

        val_epoch_loss = float("nan")
        if val_dl is not None:
            val_epoch_loss = _eval_epoch_loss(
                model=model,
                dl=val_dl,
                masker=masker,
                device=device,
                use_amp=use_amp,
                max_batches=cfg2.val_max_batches,
            )

        if val_dl is not None:
            print(
                f"epoch_end={epoch+1}/{cfg2.epochs} train_loss={train_epoch_loss:.6f} "
                f"val_loss={val_epoch_loss:.6f}"
            )
        else:
            print(f"epoch_end={epoch+1}/{cfg2.epochs} train_loss={train_epoch_loss:.6f}")

        ckpt_path = os.path.join(cfg2.save_dir, f"ckpt_epoch{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "step": step,
                "model": cfg2.model,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": cfg2.__dict__,
                "train_loss": train_epoch_loss,
                "val_loss": val_epoch_loss,
            },
            ckpt_path,
        )
        print(f"saved: {ckpt_path}")

        # Save best + early stopping (only when validation is enabled).
        if val_dl is not None and cfg2.save_best:
            improved = (best_val - val_epoch_loss) > float(cfg2.early_stop_min_delta)
            if improved:
                best_val = float(val_epoch_loss)
                best_epoch = int(epoch + 1)
                epochs_since_improve = 0
                best_path = os.path.join(cfg2.save_dir, "ckpt_best.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "model": cfg2.model,
                        "state_dict": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "config": cfg2.__dict__,
                        "train_loss": train_epoch_loss,
                        "val_loss": val_epoch_loss,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                    },
                    best_path,
                )
                print(f"saved_best: {best_path} (epoch={best_epoch} val_loss={best_val:.6f})")
            else:
                epochs_since_improve += 1

            if cfg2.early_stop_patience > 0 and epochs_since_improve >= cfg2.early_stop_patience:
                print(
                    f"early_stop: no improvement for {epochs_since_improve} epochs "
                    f"(best epoch={best_epoch} val_loss={best_val:.6f})"
                )
                break


if __name__ == "__main__":
    main()


