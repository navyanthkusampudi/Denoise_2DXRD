import os
import sys
from pathlib import Path
import unittest
from typing import Any

# Ensure repo root is importable when running from within tests/ (Windows PowerShell etc.)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Windows/conda can load multiple OpenMP runtimes (e.g., libomp + libiomp5md) via numpy/torch.
# If you hit:
#   "OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized."
# this env var is the common workaround. It must be set BEFORE importing torch.
#
# Note: This is an unsafe workaround; the "proper" fix is to ensure only one OpenMP runtime
# is present in your environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import torch  # type: ignore
    from models.dncnn import DnCNN
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DnCNN = None  # type: ignore


def count_trainable_params(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_dncnn_params(channels: int, num_of_layers: int = 17, features: int = 64) -> int:
    """
    DnCNN (as implemented in models/dncnn.py) uses:
      - Conv(in=channels, out=features, k=3, bias=False)
      - (num_of_layers - 2) blocks of [Conv(features->features, k=3, bias=False) + BN(features) + ReLU]
      - Conv(in=features, out=channels, k=3, bias=False)

    BatchNorm2d(features) has 2*features trainable parameters (gamma, beta).
    """
    if num_of_layers < 2:
        raise ValueError("num_of_layers must be >= 2")

    k2 = 3 * 3
    first_conv = features * channels * k2
    middle_conv = features * features * k2
    middle_bn = 2 * features
    last_conv = channels * features * k2

    return first_conv + (num_of_layers - 2) * (middle_conv + middle_bn) + last_conv


@unittest.skipIf(torch is None or DnCNN is None, "torch not installed")
class TestDnCNNShapeAndParamsCPU(unittest.TestCase):
    def test_output_shape_and_param_count_cpu(self) -> None:
        # Arrange (CPU)
        channels = 1
        num_of_layers = 17
        model = DnCNN(channels=channels, num_of_layers=num_of_layers).cpu().eval()

        x = torch.randn(1, channels, 512, 512, device="cpu")

        # Act
        with torch.no_grad():
            y = model(x)

        # Assert: same HxW and same channels
        self.assertEqual(tuple(y.shape), tuple(x.shape))

        # Assert: parameter count matches expected formula
        actual = count_trainable_params(model)
        expected = expected_dncnn_params(channels=channels, num_of_layers=num_of_layers, features=64)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    # Usage (CPU): run this file directly to print output size + trainable params
    if torch is None or DnCNN is None:
        raise SystemExit("torch not installed")
    channels = 1
    model = DnCNN(channels=channels, num_of_layers=17).cpu().eval()
    x = torch.randn(1, channels, 512, 512, device="cpu")
    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Trainable params: {count_trainable_params(model)}")

    unittest.main()


