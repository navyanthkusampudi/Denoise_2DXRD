# Noise2Self: Blind Denoising by Self-Supervision

This repo demonstrates a framework for blind denoising high-dimensional measurements,
as described in the [paper](https://arxiv.org/abs/1901.11365). It can be used to calibrate 
classical image denoisers and train deep neural nets; 
the same principle works on matrices of single-cell gene expression.

<img src="https://github.com/czbiohub/noise2self/blob/master/figs/hanzi_movie.gif" width="512" height="256" title="Hanzi Noise2Self">

*The result of training a U-Net to denoise a stack of noisy Chinese characters. Note that the only input is the noisy data; no ground truth is necessary.*

## Minimal denoising pipeline (what this repo keeps)

This repo has been cleaned down to a minimal, practical pipeline for denoising a detector-image stack stored in **HDF5**:

- `train_noise2self.py`: train a model from noisy data only (self-supervised)
- `scripts/denoise_h5_stack.py`: apply a trained checkpoint to an HDF5 stack
- `mask.py`, `pipeline/`, `models/`: core implementation

Dependencies are in `environment.yml`.

## How Noise2Self training works (concept)

Because the self-supervised loss is much easier to implement than the data loading, GPU management, logging, and architecture design required for handling any particular dataset, we recommend that you take any existing pipeline for your data and simply modify the training loop.

### Traditional Supervised Learning

```
for i, batch in enumerate(data_loader):
    noisy_images, clean_images = batch
    output = model(noisy_images)
    loss = loss_function(output, clean_images)
```

### Self-Supervised Learning

```
from mask import Masker
masker = Masker()
for i, batch in enumerate(data_loader):
    noisy_images, _ = batch
    input, mask = masker.mask(noisy_images, i)
    output = model(input)
    loss = loss_function(output*mask, noisy_images*mask)
```

## HDF5 stacks (.h5)

If your stack is in an HDF5 file you can train directly.

### Expected dataset format

- Dataset key: `/data` (preferred) or `/X` (fallback)
- Dataset shape:
  - `(N, H, W)` or
  - `(N0, N1, H, W)` (treated as a flattened stack of `N=N0*N1`)

### Train

Patch-based (recommended):

```
python train_noise2self.py --h5 path\to\your.h5 --max-samples 1000 --device cuda --model unet --patch 128 --batch 16 --epochs 10 --amp
```

Recommended: enable validation + save best + early stopping:

```
python train_noise2self.py --h5 path\to\your.h5 --max-samples 1000 --device cuda --model unet --patch 128 --batch 16 --epochs 200 --amp --val-frac 0.1 --save-best --early-stop-patience 10 --early-stop-min-delta 1e-5
```

Full-frame (no patch sampling):

```
python train_noise2self.py --h5 path\to\your.h5 --max-samples 1000 --device cuda --model unet --patch 0 --batch 2 --epochs 10 --amp
```

### Denoise (inference) an HDF5 stack

If you trained with `--normalize none --no-scale-integers`, you can require reconstructable “counts-space” output:

```
python scripts/denoise_h5_stack.py --h5 path\to\your.h5 --ckpt runs/noise2self/ckpt_best.pt --out path\to\denoised.h5 --device cuda --batch 32 --amp --require-reconstructable
```



