from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id: int) -> None:
    """
    Ensure each worker has a different, deterministic RNG stream.
    PyTorch sets torch.initial_seed() per-worker; we derive numpy seed from it.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(int(worker_seed))


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
) -> DataLoader:
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


