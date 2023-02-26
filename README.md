# ACOMI

This code implements the multiple clustering framework introduced in [1]. This framework combines two stages: the first stage (preprocessing) consists of feature extraction from the images using a pre-trained Vision Transformer (vit) [2],  and a 
dimensionality reduction of the representation space using a PCA, this step is implemented in Python. The second stage performs a Bayesian Non-Parametric 
multiple clustering algorithm derived from [3], this step is implement in Scala. 

![finalPipeline](https://user-images.githubusercontent.com/78457170/208433316-d6f951ca-7d91-48ae-b9ab-38c9e63673af.jpg)

### Run the preprocessing step: 

1. Download the Aloi the multiview image dataset ALOI [4] from:  https://aloi.science.uva.nl/

**N.B.** You should donwload the following files: half resolution (illumination direction - Illumination color - Viewing direction) and fusion the contents of the three folders. The resulting folder should be located in the preprocessing folder and named ALOI. 

2. Use the following command to perform the feature extraction step: 

```
python extractFeatures.py 
```

3. Use the following command to perform the dimensionality reduction step by keeping the 53 first principal components:

```
python reduceDimensionality.py 53
```

### Run the multiple clustering step: 

The build.sh script helps the user build the project using maven. The run.sh script launches the algorithm.

### References

[1] Reda Khoufache, Mohamed Djallel Dilmi, Hanene Azzag, Etienne Goffinet, and Mustapha Lebbah. Emerging properties from Bayesian Non-Parametric for multiple clustering: Application for multi-view image dataset". In workshop DLC@ICDM 2022, Nov. 28 – Dec. 1, Orlando

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2020.

[3] Etienne Goffinet, Mustapha Lebbah, Hanane Azzag, Giraldi Loïc, Anthony Coutant. Functional Non-Parametric Latent Block Model: a Multivariate Time Series Clustering Approach for Autonomous Driving Validation. Computational Statistics and Data Analysis. 2022. https://doi.org/10.1016/j.csda.2022.107565.

[4]  J. M. Geusebroek, G. J. Burghouts, and A. W. M. Smeulders. The amsterdam library of object images. International Journal of Computer Vision, 61(1):103–112, 2005.

This work is supported by [ANR France relance](https://anr.fr/fr/lanr/instruments-de-financement/plan-de-relance/), [Devoteam](https://www.devoteam.com/) and [AMIES](https://www.agence-maths-entreprises.fr/public/pages/index.html).

N.B. If you use this code, please cite the following paper: 

```
@INPROCEEDINGS{10031070,
  author={Khoufache, Reda and Dilmi, Mohamed Djallel and Azzag, Hanene and Gofinnet, Etienne and Lebbah, Mustapha},
  booktitle={2022 IEEE International Conference on Data Mining Workshops (ICDMW)}, 
  title={Emerging properties from Bayesian Non-Parametric for multiple clustering: Application for multi-view image dataset}, 
  year={2022},
  pages={31-38},
  doi={10.1109/ICDMW58026.2022.00013}}


```
