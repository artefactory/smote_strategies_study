## Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants.

Repository for [Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants](https://arxiv.org/pdf/2402.03819.pdf) paper.

In praticular, you will find code to reproduce the paper experiments. **If you want to use our proposed mthods, please use the following updated repository** : https://github.com/artefactory/mgs-grf
## ⭐ Table of Contents
  - [Getting Started](#getting-started)
  - [Data sets](#data-sets)
  - [Acknowledgements](#acknowledgements)

## ⭐ Getting Started

If you want to reproduce our paper experiments:
  - the notebooks [here](notebooks/classif_experiments.ipynb) and [here](notebooks/distances_experiments.ipynb) reproduce the experiments
  - thise [code](./validation) contains implementation the protocols used for the numerical experiments of our article. 

In order to use our MGS strategy:
  - this [notebook](notebooks/resampling_example.ipynb) illustrates how to use it
  - the strategy is implemented [here](./oversampling_strategies/)

## ⭐ Data sets

The data sets of used for our article should be dowloaded  inside the *data/externals* folder. The data sets are available at the followings adresses :

* [Pima](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* Phoneme : https://github.com/jbrownlee/Datasets/blob/master/phoneme.csv 
* Abalone : https://archive.ics.uci.edu/dataset/1/abalone
* Wine : https://archive.ics.uci.edu/dataset/186/wine+quality
* Haberman : https://archive.ics.uci.edu/dataset/43/haberman+s+survival
* Yeast : https://archive.ics.uci.edu/dataset/110/yeast
* Vehicle : https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes
* Ionosphere : https://archive.ics.uci.edu/dataset/52/ionosphere
* Breast cancer Wisconsin : https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
* CreditCard : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* MagicTel : https://www.openml.org/d/44125
* California : https://www.openml.org/d/44090
* House_16H : https://openml.org/d/821 


## ⭐ Acknowledgements

This work was done through a partenership between **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.

<p align="center">
  <a href="https://www.artefact.com/data-consulting-transformation/artefact-research-center/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_arc.png" height="80" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.lpsm.paris/">
    <img src="data/logos//logo_LPSM.jpg" height="95" />
  </a>
</p>

If you find the code usefull, please consider citing us :
```bib
@article{sakho2024we,
  title={Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants},
  author={Sakho, Abdoulaye and Malherbe, Emmanuel and Scornet, Erwan},
  journal={arXiv preprint arXiv:2402.03819},
  year={2024}
}
```