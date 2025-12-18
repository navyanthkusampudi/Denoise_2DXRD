"""
Masking utilities for Noise2Self self-supervised denoising.

This module implements the masking strategy described in the Noise2Self paper,
where pixels are masked out during training so the network learns to predict
pixel values from their surrounding context (J-invariant functions).

The key idea: by hiding a pixel from the network and asking it to predict that
pixel's value from its neighbors, we can train a denoiser without clean targets.
"""

import numpy as np
import torch


class Masker():
    """
    Object for masking and demasking images in Noise2Self training.
    
    The masker creates a grid-based masking pattern where pixels at regular
    intervals are masked out. During training, these masked pixels serve as
    targets while the surrounding pixels provide context for prediction.
    
    Attributes:
        grid_size (int): Spacing between masked pixels (width x width grid).
        n_masks (int): Total number of unique mask patterns (grid_size^2).
        mode (str): How to handle masked pixels - 'zero' or 'interpolate'.
        infer_single_pass (bool): If True, use full image for inference.
        include_mask_as_input (bool): If True, concatenate mask to input.
    """

    def __init__(self, width=3, mode='zero', infer_single_pass=False, include_mask_as_input=False):
        """
        Initialize the Masker.
        
        Args:
            width (int): Grid spacing. A width of 3 means every 3rd pixel 
                        (in both dimensions) will be masked, creating 9 
                        unique mask patterns.
            mode (str): Masking mode:
                - 'zero': Replace masked pixels with zeros.
                - 'interpolate': Replace masked pixels with interpolated 
                                 values from neighbors.
            infer_single_pass (bool): If True, run inference on full image
                                      without masking. If False, aggregate
                                      predictions from all mask patterns.
            include_mask_as_input (bool): If True, append the binary mask
                                          as an additional input channel.
        """
        self.grid_size = width
        self.n_masks = width ** 2  # Total unique mask patterns

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i):
        """
        Apply the i-th mask pattern to input tensor X.
        
        The mask pattern is determined by the phase offset (i) which controls
        which pixels in the grid are masked. For a 3x3 grid, there are 9 
        unique patterns (i = 0 to 8), each masking different pixel positions.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch, channels, height, width).
            i (int): Mask pattern index (0 to n_masks-1).
            
        Returns:
            tuple: (net_input, mask)
                - net_input: Masked input tensor for the network.
                - mask: Binary mask tensor indicating which pixels were masked (1=masked).
        """
        # Calculate phase offsets from mask index
        # This determines which pixels in the repeating grid pattern are masked
        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size
        
        # Generate binary mask where 1 = pixel to be predicted (held out)
        mask = pixel_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
        mask = mask.to(X.device)

        # Inverse mask: 1 = pixels available as context
        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        # Apply masking strategy
        if self.mode == 'interpolate':
            # Replace masked pixels with interpolated values from neighbors
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            # Replace masked pixels with zeros
            masked = X * mask_inv
        else:
            raise NotImplementedError
        
        # Optionally include mask as additional input channel
        if self.include_mask_as_input:
            net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        """Return the number of unique mask patterns."""
        return self.n_masks

    def infer_full_image(self, X, model):
        """
        Denoise a full image using the trained model.
        
        Two inference modes are supported:
        1. Single pass: Run the full image through the network directly.
        2. Multi-pass: Apply all mask patterns and aggregate predictions,
           ensuring each pixel is predicted from its neighbors only.
        
        Args:
            X (torch.Tensor): Noisy input image of shape (batch, channels, H, W).
            model: Trained denoising network.
            
        Returns:
            torch.Tensor: Denoised image of the same shape as input.
        """
        if self.infer_single_pass:
            # Single pass inference - use full image directly
            if self.include_mask_as_input:
                # Append zeros as mask channel (indicating no pixels are masked)
                net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            # Multi-pass inference - aggregate predictions from all mask patterns
            # This ensures each pixel is predicted using only neighbor information
            net_input, mask = self.mask(X, 0)
            net_output = model(net_input)

            # Accumulator for aggregating predictions
            acc_tensor = torch.zeros(net_output.shape).cpu()

            # Apply each mask pattern and collect predictions for masked pixels only
            for i in range(self.n_masks):
                net_input, mask = self.mask(X, i)
                net_output = model(net_input)
                # Only keep predictions for the masked pixels (where mask=1)
                acc_tensor = acc_tensor + (net_output * mask).cpu()

            return acc_tensor


def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    """
    Create a binary mask with 1s at regular grid positions.
    
    Generates a mask where pixels at positions (i, j) are set to 1 if:
        i % patch_size == phase_x AND j % patch_size == phase_y
    
    This creates a sparse grid pattern where masked pixels are evenly
    distributed across the image.
    
    Args:
        shape (tuple): Shape of the mask (height, width).
        patch_size (int): Grid spacing between masked pixels.
        phase_x (int): Vertical offset (0 to patch_size-1).
        phase_y (int): Horizontal offset (0 to patch_size-1).
        
    Returns:
        torch.Tensor: Binary mask of shape (height, width) with 1s at grid positions.
        
    Example:
        For patch_size=3, phase_x=0, phase_y=0, masked pixels (*) are:
        * . . * . . * . .
        . . . . . . . . .
        . . . . . . . . .
        * . . * . . * . .
        ...
    """
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    return torch.Tensor(A)


def interpolate_mask(tensor, mask, mask_inv):
    """
    Replace masked pixels with interpolated values from neighbors.
    
    Uses a weighted average of neighboring pixels (excluding the center)
    to fill in masked positions. This can provide better context than
    zeros for the network.
    
    The interpolation kernel:
        [0.5, 1.0, 0.5]
        [1.0, 0.0, 1.0]  <- center is 0 (excluded from average)
        [0.5, 1.0, 0.5]
    
    Args:
        tensor (torch.Tensor): Input image tensor (batch, channels, H, W).
        mask (torch.Tensor): Binary mask where 1 = pixels to interpolate.
        mask_inv (torch.Tensor): Inverse mask where 1 = original pixels to keep.
        
    Returns:
        torch.Tensor: Image with masked pixels replaced by interpolated values.
    """
    device = tensor.device

    mask = mask.to(device)

    # Interpolation kernel - weights for neighboring pixels
    # Center pixel has weight 0 (we don't use the pixel being interpolated)
    # Cardinal neighbors (up/down/left/right) have weight 1.0
    # Diagonal neighbors have weight 0.5
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, 3, 3)
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()  # Normalize to sum to 1

    # Apply convolution to compute weighted average of neighbors
    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    # Combine: use interpolated values where mask=1, original values elsewhere
    return filtered_tensor * mask + tensor * mask_inv
