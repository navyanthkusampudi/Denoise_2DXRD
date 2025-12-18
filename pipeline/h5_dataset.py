from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except Exception as e:  # pragma: no cover
    h5py = None


Size2D = Union[int, Tuple[int, int]]

DEFAULT_H5_KEY = "/img"
# Historical default used in earlier experiments. We no longer hard-enforce this
# so the same pipeline can train on cropped detector frames (e.g. 400x490).
EXPECTED_HW = (512, 512)


@dataclass(frozen=True)
class StackSpec:
    """A thin description of a detector-image stack."""

    n: int
    h: int
    w: int
    dtype: Any
    grid: Optional[Tuple[int, int]] = None  # (n0, n1) when stored as (n0,n1,H,W)


def _as_hw(size: Optional[Size2D]) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, tuple) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Invalid size: {size!r} (expected int or (h,w) tuple)")


def _infer_stack_spec_from_shape(shape: Tuple[int, ...], dtype: Any) -> StackSpec:
    """
    Infer stack description from an HDF5 dataset shape.

    Supported layouts:
      - (N, H, W)
      - (N0, N1, H, W)  (flattened logically to N=N0*N1)
    """
    if len(shape) == 3:
        n, h, w = shape
        return StackSpec(n=int(n), h=int(h), w=int(w), dtype=dtype, grid=None)

    if len(shape) == 4:
        n0, n1, h, w = shape
        n = int(n0) * int(n1)
        return StackSpec(n=n, h=int(h), w=int(w), dtype=dtype, grid=(int(n0), int(n1)))

    raise ValueError(f"Unsupported stack ndim={len(shape)}. Expected 3D or 4D dataset.")


def _to_float32(x: np.ndarray, scale_integers: bool = True) -> np.ndarray:
    """Convert to float32 and optionally scale integer types to [0,1]."""
    if np.issubdtype(x.dtype, np.integer) and scale_integers:
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / float(info.max)
        return x
    return x.astype(np.float32, copy=False)


def _normalize_patch(
    x: np.ndarray,
    mode: str,
    eps: float = 1e-6,
    pmin: float = 1.0,
    pmax: float = 99.0,
) -> np.ndarray:
    """
    Normalize a (C,H,W) float32 patch.
    Modes:
      - 'none'
      - 'minmax'      : (x - min)/(max-min)
      - 'percentile'  : robust min/max by percentiles (per-sample, across all channels)
    """
    mode = mode.lower()
    if mode == "none":
        return x

    if mode == "minmax":
        mn = float(x.min())
        mx = float(x.max())
        return (x - mn) / (mx - mn + eps)

    if mode == "percentile":
        lo = float(np.percentile(x, pmin))
        hi = float(np.percentile(x, pmax))
        return (x - lo) / (hi - lo + eps)

    raise ValueError(f"Unknown normalize mode: {mode!r}")


def _random_crop(rng: np.random.Generator, x: np.ndarray, patch_hw: Tuple[int, int]) -> np.ndarray:
    """
    x: (C,H,W) numpy
    returns: (C,ph,pw) numpy
    """
    ph, pw = patch_hw
    _, h, w = x.shape
    if ph > h or pw > w:
        raise ValueError(f"Patch {patch_hw} larger than image {(h, w)}.")
    top = int(rng.integers(0, h - ph + 1))
    left = int(rng.integers(0, w - pw + 1))
    return x[:, top : top + ph, left : left + pw]


def _augment_90(rng: np.random.Generator, x: np.ndarray, enable: bool) -> np.ndarray:
    """Random flips + 90-deg rotations on (C,H,W)."""
    if not enable:
        return x
    k = int(rng.integers(0, 4))
    x = np.rot90(x, k=k, axes=(1, 2))
    if bool(rng.integers(0, 2)):
        x = x[:, :, ::-1]  # horizontal flip
    if bool(rng.integers(0, 2)):
        x = x[:, ::-1, :]  # vertical flip
    return np.ascontiguousarray(x)


def _get_stack_dataset(f: "h5py.File") -> "h5py.Dataset":
    """
    Fetch the detector stack dataset.

    Preferred key is '/data' (as shown in HDFView). For backward compatibility we
    also accept '/X' if present.
    """
    candidates = [DEFAULT_H5_KEY, DEFAULT_H5_KEY.lstrip("/"), "/X", "X", "data"]
    last_err: Exception | None = None
    for k in candidates:
        try:
            return f[k]
        except Exception as e:
            last_err = e
            continue
    raise KeyError(str(last_err) if last_err is not None else "HDF5 dataset not found (expected '/data' or '/X').")


class H5StackDataset(Dataset):
    """
    Dataset for a stack of detector images stored in an HDF5 dataset.

    Expected:
      - dataset key: '/data' (preferred) or '/X' (fallback)
      - dataset shape: (N,512,512) or (N0,N1,512,512)

    Notes:
      - Uses lazy opening per-process. Each DataLoader worker will open its own file handle.
      - Keeps all Noise2Self masking logic OUT of the dataset (same as NpyStackDataset).
    """

    def __init__(
        self,
        h5_path: str,
        *,
        patch_size: Optional[Size2D] = None,
        normalize: str = "percentile",
        scale_integers: bool = True,
        augment: bool = False,
        seed: int = 0,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        if h5py is None:
            raise ImportError(
                "h5py is required to use H5StackDataset. Install it (pip/conda) or convert to .npy."
            )

        self.h5_path = h5_path

        # Inspect shape/dtype once.
        with h5py.File(self.h5_path, "r") as f:
            dset = _get_stack_dataset(f)
            self.spec = _infer_stack_spec_from_shape(tuple(dset.shape), dset.dtype)

        self.patch_hw = _as_hw(patch_size)
        self.normalize = normalize
        self.scale_integers = scale_integers
        self.augment = augment

        self.base_seed = int(seed)

        if indices is None:
            self.indices = np.arange(self.spec.n, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        # Lazy handles (per-process)
        self._h5 = None
        self._dset = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _ensure_open(self) -> None:
        if self._h5 is not None and self._dset is not None:
            return
        # Each worker process will create its own handle.
        self._h5 = h5py.File(self.h5_path, "r")
        self._dset = _get_stack_dataset(self._h5)

    def _get_frame_numpy(self, frame_idx: int) -> np.ndarray:
        self._ensure_open()
        dset = self._dset
        assert dset is not None
        if dset.ndim == 3:
            x = dset[frame_idx]  # (H,W)
        else:
            assert self.spec.grid is not None
            n0, n1 = self.spec.grid
            i = int(frame_idx // n1)
            j = int(frame_idx % n1)
            x = dset[i, j]  # (H,W)
        x = x[None, ...]  # (1,H,W)
        return np.asarray(x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        frame_idx = int(self.indices[idx])
        x = self._get_frame_numpy(frame_idx)  # (C,H,W) numpy

        # Per-item RNG that is stable under multi-worker shuffling.
        seed = (self.base_seed + 1_000_003 * frame_idx + 97 * idx) % (2**32 - 1)
        rng = np.random.default_rng(int(seed))

        x = _to_float32(x, scale_integers=self.scale_integers)

        if self.patch_hw is not None:
            x = _random_crop(rng, x, self.patch_hw)

        x = _augment_90(rng, x, self.augment)
        x = _normalize_patch(x, self.normalize)

        x = np.ascontiguousarray(x, dtype=np.float32)
        meta: Dict[str, Any] = {"frame_idx": frame_idx, "shape": tuple(x.shape)}
        if self.spec.grid is not None:
            n0, n1 = self.spec.grid
            meta["grid"] = (n0, n1)
            meta["grid_idx"] = (int(frame_idx // n1), int(frame_idx % n1))
        return torch.from_numpy(x), meta


def list_h5_datasets(h5_path: str) -> list[tuple[str, tuple[int, ...], str]]:
    """
    Return a list of (key, shape, dtype_str) for all HDF5 datasets in a file.
    Useful for figuring out the correct dataset key (--h5-key).
    """
    if h5py is None:
        raise ImportError("h5py is required to inspect HDF5 files.")

    out: list[tuple[str, tuple[int, ...], str]] = []
    with h5py.File(h5_path, "r") as f:
        def visitor(name: str, obj: Any) -> None:
            try:
                if isinstance(obj, h5py.Dataset):
                    out.append(("/" + name.lstrip("/"), tuple(int(x) for x in obj.shape), str(obj.dtype)))
            except Exception:
                # Skip objects that error during inspection
                return

        f.visititems(visitor)
    return out


