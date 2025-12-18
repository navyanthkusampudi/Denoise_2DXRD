# Denoise_2DXRD
ML models to denoise 2D XRD images in resiprocal space

### Data used for machine learning

The machine learning pipeline uses preprocessed 2D X-ray diffraction (XRD) images of size 512 × 512 stored as float32 arrays. 
The data originate from a 141 × 121 real-space scan grid, with one diffraction image per scan position.

Prior to machine learning, the following preprocessing steps were applied: 

the detector images were cropped to the left 512 × 512 pixel region, 
invalid detector pixels marked by the sentinel value (2³² − 1) were replaced with zero, 
and a variance-stabilizing transform was applied to the intensities,
\[
I \rightarrow \sqrt{\max(I, 0) + 0.375}.
\]


The preprocessed data are reshaped into individual images with an added channel dimension for training. 
During Noise2Self training, random pixels are temporarily masked on-the-fly to define the self-supervised loss; 
this masking is used only during training and does not permanently modify the data.


### Why Noise2Self is used for this XRD data

The 2D X-ray diffraction images in this dataset are affected by photon-counting (Poisson) noise, detector noise, and occasional defective pixels, while the underlying diffraction signal (rings, peaks, and diffuse scattering) is spatially correlated across neighboring detector pixels. In this setting, clean “ground-truth” diffraction images are not available, making supervised denoising impractical.

Noise2Self is a self-supervised denoising method that does not require clean reference data. It exploits the fact that noise in individual detector pixels is approximately independent, whereas the diffraction signal is correlated across neighboring pixels. By masking individual pixels and training a neural network to predict their values from surrounding pixels, the network learns to reconstruct the underlying diffraction signal while suppressing uncorrelated noise.

Because the method does not use paired noisy–clean data, it is well suited for experimental XRD measurements where repeated acquisitions or noise-free references are unavailable.

### Noise2Self masking principle

During Noise2Self training, a subset of pixels in each diffraction image is randomly masked (set to zero) and treated as unknown. A convolutional neural network is trained to predict the values of these masked pixels using only the surrounding, unmasked pixels. The training loss is computed exclusively on the masked pixels, ensuring that the network cannot trivially copy the input and instead learns to reconstruct the underlying signal from local spatial context. This exploits the spatial correlation of diffraction features and the approximate pixel-wise independence of noise.

### Inspiration
Our model is inspired by the **Noise2Self** self-supervised denoising framework:
> **Noise2Self: Blind Denoising by Self-Supervision**  
> Joshua Batson, Loïc Royer  
> ICML 2019

Noise2Self introduces a self-supervised denoising principle that enables learning from single noisy images without clean targets.

🔗 Paper: https://arxiv.org/abs/1901.11365


