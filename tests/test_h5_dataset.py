import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Ensure repo root is importable when running from within tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


@unittest.skipIf(torch is None, "torch not installed")
@unittest.skipIf(h5py is None, "h5py not installed")
class TestH5DatasetXOnly(unittest.TestCase):
    def test_reads_x_stack(self) -> None:
        from pipeline.h5_dataset import H5StackDataset, EXPECTED_HW, list_h5_datasets
        with tempfile.TemporaryDirectory() as td:
            h5_path = os.path.join(td, "one.h5")
            with h5py.File(h5_path, "w") as f:
                n0, n1 = 2, 3
                h, w = EXPECTED_HW
                f.create_dataset("data", data=np.zeros((n0, n1, h, w), dtype=np.float16))

            # list_h5_datasets reports absolute keys
            listed = list_h5_datasets(h5_path)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0][0], "/data")

            ds = H5StackDataset(h5_path, patch_size=None, normalize="none", augment=False, seed=0)
            self.assertEqual(len(ds), n0 * n1)
            x0, _m0 = ds[0]
            self.assertEqual(tuple(x0.shape), (1, h, w))


