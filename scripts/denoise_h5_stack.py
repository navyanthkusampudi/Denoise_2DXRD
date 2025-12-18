from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover
    h5py = None

# ---- Ensure repo root is importable (so `import mask` works when running from runs/reconst/) ----
_repo = None
for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (p / "mask.py").exists() and (p / "train_noise2self.py").exists():
        _repo = p
        break
if _repo is None:
    raise RuntimeError("Could not find repo root (expected to find mask.py and train_noise2self.py).")
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from mask import Masker
from models.unet import Unet
from models.babyunet import BabyUnet
from models.dncnn import DnCNN
from models.singleconv import SingleConvolution


def _get_h5_dataset(f: "h5py.File") -> "h5py.Dataset":
    for k in ["img", "/img","data", "/data", "X", "/X"]:
        if k in f:
            return f[k]
    raise KeyError(f"Could not find dataset key in H5. Keys: {list(f.keys())}")


def _pad_bhw_to_multiple(x: np.ndarray, mult: int = 16) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pad (B,H,W) to H,W divisible by mult. Returns (x_pad, pads)."""
    assert x.ndim == 3
    b, h, w = x.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x_pad = np.pad(
        x,
        ((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )
    return x_pad, (pad_top, pad_bot, pad_left, pad_right)


def _unpad_bhw(x_pad: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    """Unpad (B,H,W) given pads=(top,bot,left,right)."""
    assert x_pad.ndim == 3
    pad_top, pad_bot, pad_left, pad_right = pads
    if pad_top == pad_bot == pad_left == pad_right == 0:
        return x_pad
    b, h, w = x_pad.shape
    return x_pad[:, pad_top : h - pad_bot, pad_left : w - pad_right]


def _normalize_2d(x: np.ndarray, mode: str, *, eps: float = 1e-6, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "none":
        return x
    if mode == "minmax":
        mn = float(np.min(x))
        mx = float(np.max(x))
        return (x - mn) / (mx - mn + eps)
    if mode == "percentile":
        lo = float(np.percentile(x, pmin))
        hi = float(np.percentile(x, pmax))
        return (x - lo) / (hi - lo + eps)
    raise ValueError(f"Unknown normalize mode: {mode!r}")


def _build_model(model_name: str, in_ch: int, out_ch: int) -> torch.nn.Module:
    name = str(model_name).lower()
    if name == "unet":
        return Unet(in_ch, out_ch)
    if name in {"babyunet", "baby-unet"}:
        return BabyUnet(in_ch, out_ch)
    if name == "dncnn":
        # Project DnCNN API differs across versions; this repo’s DnCNN supports (in_ch,out_ch) in train script,
        # but notebooks sometimes call DnCNN(channels=...).
        try:
            return DnCNN(in_ch, out_ch)  # type: ignore[arg-type]
        except Exception:
            return DnCNN(channels=in_ch)  # type: ignore[call-arg]
    if name in {"conv", "convolution", "singleconv"}:
        return SingleConvolution(in_ch, out_ch, width=3)
    raise ValueError(f"Unknown model: {model_name!r}")


@dataclass(frozen=True)
class InferenceCfg:
    h5_in: str
    ckpt: str
    h5_out: str
    batch: int
    device: str
    amp: bool
    require_reconstructable: bool


def parse_args() -> InferenceCfg:
    p = argparse.ArgumentParser(description="Denoise an HDF5 detector stack using a trained Noise2Self checkpoint.")
    p.add_argument("--h5", required=True, help="Input HDF5 path (expects dataset /data or /X).")
    p.add_argument("--ckpt", required=True, help="Checkpoint path (ckpt_best.pt or ckpt_epoch*.pt).")
    p.add_argument("--out", required=True, help="Output HDF5 path to write (float32 counts).")
    p.add_argument("--batch", type=int, default=32, help="Inference batch size.")
    p.add_argument("--device", default="cuda", help="cuda|cuda:0|cpu")
    p.add_argument("--amp", action="store_true", help="Use autocast on CUDA.")
    p.add_argument(
        "--require-reconstructable",
        action="store_true",
        help="Fail unless ckpt was trained with normalize='none' (so output is meaningful counts).",
    )
    a = p.parse_args()
    return InferenceCfg(
        h5_in=str(a.h5),
        ckpt=str(a.ckpt),
        h5_out=str(a.out),
        batch=int(a.batch),
        device=str(a.device),
        amp=bool(a.amp),
        require_reconstructable=bool(a.require_reconstructable),
    )


def main() -> None:
    if h5py is None:
        raise ImportError("h5py is required for this script.")

    cfg = parse_args()

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = bool(cfg.amp and device.type == "cuda")

    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    ckpt_cfg: dict[str, Any] = ckpt.get("config", {}) or {}

    model_name = str(ckpt_cfg.get("model", "unet"))
    in_ch = int(ckpt_cfg.get("in_channels", 1))
    out_ch = int(ckpt_cfg.get("out_channels", 1))
    include_mask = bool(ckpt_cfg.get("include_mask_as_input", False))
    masker_width = int(ckpt_cfg.get("masker_width", 4))
    masker_mode = str(ckpt_cfg.get("masker_mode", "interpolate"))
    normalize_mode = str(ckpt_cfg.get("normalize", "percentile")).lower()
    scale_integers = bool(ckpt_cfg.get("scale_integers", True))

    if cfg.require_reconstructable and normalize_mode != "none":
        raise ValueError(
            f"Checkpoint normalize={normalize_mode!r}. This is not uniquely invertible to counts. "
            f"Use a reconst checkpoint trained with --normalize none."
        )

    model = _build_model(model_name, in_ch, out_ch)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    masker = Masker(
        width=masker_width,
        mode=masker_mode,
        infer_single_pass=True,
        include_mask_as_input=include_mask,
    )

    print("device:", device)
    print("h5_in:", cfg.h5_in)
    print("ckpt:", cfg.ckpt)
    print("h5_out:", cfg.h5_out)
    print("model:", model_name, "in_ch=", in_ch, "out_ch=", out_ch)
    print("masker:", f"width={masker_width}", f"mode={masker_mode}", f"include_mask={include_mask}")
    print("ckpt scaling:", f"normalize={normalize_mode}", f"scale_integers={scale_integers}")
    print("infer:", f"batch={cfg.batch}", f"amp={use_amp}")

    os.makedirs(os.path.dirname(cfg.h5_out) or ".", exist_ok=True)

    with h5py.File(cfg.h5_in, "r") as fin, h5py.File(cfg.h5_out, "w") as fout:
        ds_in = _get_h5_dataset(fin)
        if ds_in.ndim != 4:
            raise ValueError(f"Expected (sy,sx,H,W) dataset, got shape={ds_in.shape}")
        sy, sx, h, w = (int(x) for x in ds_in.shape)
        in_dtype = ds_in.dtype

        ds_out = fout.create_dataset(
            "data",
            shape=(sy, sx, h, w),
            dtype=np.float32,
            chunks=(1, 1, h, w),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        meta = fout.create_group("meta")
        meta.attrs["input_h5"] = str(cfg.h5_in)
        meta.attrs["input_dataset"] = str(ds_in.name)
        meta.attrs["checkpoint"] = str(cfg.ckpt)
        meta.attrs["model"] = str(model_name)
        meta.attrs["normalize"] = str(normalize_mode)
        meta.attrs["scale_integers"] = bool(scale_integers)
        meta.attrs["input_dtype"] = str(in_dtype)
        meta.attrs["note"] = "data is denoised output in float32 counts when normalize='none'."

        # scale factor for integer-coded detector counts
        counts_scale = 1.0
        if np.issubdtype(in_dtype, np.integer) and scale_integers:
            counts_scale = float(np.iinfo(in_dtype).max)

        # Process one scan row at a time to reduce HDF5 overhead.
        for yy in range(sy):
            row = np.asarray(ds_in[yy, :, :, :], dtype=np.float32)  # (sx,H,W) counts

            # Convert to model space (fast path for reconst counts-space training)
            if normalize_mode == "none" and (not scale_integers):
                row_model = row
            else:
                if scale_integers and np.issubdtype(in_dtype, np.integer):
                    row01 = row / counts_scale
                else:
                    row01 = row
                row_model = np.stack([_normalize_2d(row01[i], normalize_mode) for i in range(row01.shape[0])], axis=0)

            # Inference in batches
            den_row_counts = np.empty_like(row, dtype=np.float32)
            for x0 in range(0, sx, cfg.batch):
                x1 = min(sx, x0 + cfg.batch)
                x_bhw = row_model[x0:x1]  # (B,H,W)
                x_pad, pads = _pad_bhw_to_multiple(x_bhw, 16)
                xt = torch.from_numpy(x_pad[:, None, ...]).to(device)  # (B,1,H,W)
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        yt = masker.infer_full_image(xt, model)
                y = yt.detach().to("cpu").float().numpy()[:, 0]  # (B,H,W) model space
                y = _unpad_bhw(y, pads)

                # Back to counts if reconstructable; otherwise best effort.
                if normalize_mode == "none":
                    if scale_integers and np.issubdtype(in_dtype, np.integer):
                        y_counts = y * counts_scale
                    else:
                        y_counts = y
                else:
                    # No unique inverse; store model-space-ish scaled to counts factor for convenience
                    y_counts = y * counts_scale

                den_row_counts[x0:x1] = np.asarray(y_counts, dtype=np.float32)

            # Write row
            ds_out[yy, :, :, :] = den_row_counts

            if (yy % 10) == 0 or yy == sy - 1:
                print(f"row {yy+1}/{sy} done")

        print("Saved:", cfg.h5_out)


if __name__ == "__main__":
    main()


