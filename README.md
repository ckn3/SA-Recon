# shape-adaptive-reconstruction


This code is an implementation of "Classification of Hyperspectral Images Using SVM with Shape-adaptive Reconstruction and Smoothed Total Variation" [Link](https://arxiv.org/abs/2203.15619). The code can be used as a denoising method, which is a pre-processing step for multispectral and hyperspectral datasets with applications to land cover classification. 

The code uses several Matlab Toolboxes, see [LASIP](https://webpages.tuni.fi/lasip/2D/) and [SA-DCT](https://webpages.tuni.fi/foi/SA-DCT/).

Notes:
- The code only contains the reconstruction part that utilize the spatial information. You could apply any other methods on reconstructed datasets. Contact me via email if you need more info about the code.
- To run a demo, please run the SaR_main.m. Make sure you download the benchmark datasets before testing.
- To apply the code on your dataset, you could simply change the input datasets.
- Contact: kangnicui2@gmail.com

## References
**Li, R., Cui, K., Chan, R. H., & Plemmons, R. J.**. "Classification of Hyperspectral Images Using SVM with Shape-adaptive Reconstruction and Smoothed Total Variation". in *Proc IEEE Int Geosci Remote Sens Symp*, IEEE, 2022.
