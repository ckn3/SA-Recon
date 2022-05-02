# shape-adaptive-reconstruction


This code is an implementation of Shape-adaptive Reconstruction (SaR) proposed in "Classification of Hyperspectral Images Using SVM with Shape-adaptive Reconstruction and Smoothed Total Variation", [Link](https://arxiv.org/abs/2203.15619). 

The SaR code can be used as a denoising method for hyperspectral datasets. In [this paper](https://arxiv.org/abs/2203.15619), SaR is firstly used as a preprocessing step before training a semi-supervised classifier. SaR has been applied in an unsupervised diffusion-based algorithm called [DSIRC](https://arxiv.org/abs/2204.13497) as a smoothing stage as well, demonstrating the efficacy of SaR.

SaR uses several Matlab Toolboxes, see [LASIP](https://webpages.tuni.fi/lasip/2D/) and [SA-DCT](https://webpages.tuni.fi/foi/SA-DCT/).

Notes:
- The code only contains the reconstruction part that utilize the spatial information. You could apply any other methods on reconstructed datasets. Contact me via email if you need more info about the code.
- To run a demo of SaR, please run SaR_main.m. Make sure you download the benchmark datasets before testing.
- To run a demo of DSIRC, please run DSIRCGS.m (with [D-VIC toolbox](https://github.com/sampolk/D-VIC)). 
- To apply the code on your dataset, you could simply change the input datasets.
- Contact: kangnicui2@gmail.com

*If you find it useful or use it in any publications, please cite the following papers:*
## References

**Li, R., Cui, K., Chan, R. H., & Plemmons, R. J.**. "Classification of Hyperspectral Images Using SVM with Shape-adaptive Reconstruction and Smoothed Total Variation". in *Proc IEEE Int Geosci Remote Sens Symp*, IEEE, 2022. [Link](https://arxiv.org/abs/2203.15619).

**Cui, K., Li, R., Polk, S.L., Murphy, J.M., & Plemmons, R. J., Chan, R. H.**. "Unsupervised Spatial-spectral Hyperspectral Image Reconstruction and Clustering with Diffusion Geometry". in *arXiv*, 2022. [Link](https://arxiv.org/abs/2204.13497).
