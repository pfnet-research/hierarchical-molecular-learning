This is implementation of "Semi-supervised learning of hierarchical representations of molecules using neural message passing"
presetented at NIPS2017 workshop on Machine Learning for Molecules and Materials.


# Dependency

* Chainer (<=3.1.0)
* NumPy
* SciPy
* scikit-learn
* six

You can install these packages with pip by `pip install -r requirements.txt`
or create a conda environment with these packages installed by `conda env create -n <env name> --file env.yaml`.

We confirm the code with following environment.

```
chainer==3.1.0
numpy==1.13.3
scikit-learn==0.19.1
scipy==1.0.1
six==1.10.0
```

Note that this code does not work with Chainer newer than v3.1.0 due to changes made in Chainer.
We will solve the problem by fixing Chainer itself.
See [chainer/chainer#4877](https://github.com/chainer/chainer/issues/4877) for detail.


# Usage

```python
cd unsupNFP
python train.py mutag  # Use the MUTAG dataset
python train.py ptc # Use the PTC dataset
```

This repository has code for the experiments of unsupervised setting only.
Code for the semi-supervised setting is under preparation.

# Data source

* MUTAG: [BorgwardtLab/GraphKernels](https://github.com/BorgwardtLab/GraphKernels)
* PTC: [predictive-toxicology.org](https://www.predictive-toxicology.org)
* malaria: [HIPS/neural-fingerprint](https://github.com/HIPS/neural-fingerprint)

# Reference

Nguyen, H., Maeda, S. I., & Oono, K. (2017). Semi-supervised learning of hierarchical representations of molecules using neural message passing. arXiv preprint arXiv:1711.10168 [URL](https://arxiv.org/abs/1711.10168).


# Contact

Kenta Oono (oono@preferred.jp)
