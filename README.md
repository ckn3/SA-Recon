# shape-adaptive-reconstruction


This code is an implementation of **Shape-adaptive Reconstruction (SaR)** proposed in "Classification of Hyperspectral Images Using SVM with Shape-adaptive Reconstruction and Smoothed Total Variation", see [Link](https://arxiv.org/abs/2203.15619), and "Unsupervised Spatial-spectral Hyperspectral Image Reconstruction and Clustering with Diffusion Geometry", see [Link](https://arxiv.org/abs/2204.13497). 

The SaR code can be used as a denoising method for remote sensing datasets. In [this paper](https://arxiv.org/abs/2203.15619), SaR is firstly used as a preprocessing step before training a semi-supervised classifier. SaR has been applied in an unsupervised diffusion-based algorithm called [DSIRC](https://arxiv.org/abs/2204.13497) as a smoothing stage as well.

SaR uses several Matlab Toolboxes, such as [LASIP](https://webpages.tuni.fi/lasip/2D/) and [SA-DCT](https://webpages.tuni.fi/foi/SA-DCT/). SVM-STV uses the [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) toolbox to implement SVMs. DSIRC uses the [D-VIC](https://github.com/sampolk/D-VIC) toolbox for unsupervised diffusion learning.

Notes:
- The code contains the Shape-adaptive Reconstruction part that use the spatial information to reduce the noise.
- To run a demo of SaR, please run SaR_main.m. Make sure you download the benchmark datasets before trying.
- To run a demo of SaR-SVM-STV (semi-supervised), please run SaR_SVM_STV.m. 
- To run a demo of DSIRC (unsupervised), please run DSIRCGS.m. 
- To apply the code on your dataset, you could simply change the input datasets.
- Contact: kangnicui2@gmail.com

*If you find it useful or use it in any publications, please cite the following papers:*
## References

**Li, R., Cui, K., Chan, R. H., & Plemmons, R. J.**. "Classification of Hyperspectral Images Using SVM with Shape-adaptive Reconstruction and Smoothed Total Variation". in *Proc IEEE Int Geosci Remote Sens Symp*, IEEE, 2022. [Link](https://arxiv.org/abs/2203.15619).

**Cui, K., Li, R., Polk, S.L., Murphy, J.M., & Plemmons, R. J., Chan, R. H.**. "Unsupervised Spatial-spectral Hyperspectral Image Reconstruction and Clustering with Diffusion Geometry". in *Proc IEEE Workshop Hyperspectral Image Signal Process Evol Remote Sens*, IEEE, 2022. [Link](https://arxiv.org/abs/2204.13497).
