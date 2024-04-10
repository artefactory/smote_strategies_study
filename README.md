## SMOTE strategies study

Repository for [Theoretical and experimental study of SMOTE](https://arxiv.org/pdf/2402.03819.pdf).

In praticular, you will find code to reproduce the paper experiments as well as an nice implementation of our *new* and *efficient* strategy for your projects.


## Table of Contents
  - [Getting Started](#getting-started)
  - [Data sets](#data-sets)
  - [Acknowledgements](#acknowledgements)

## Getting Started

In order to use our xxx strategy:
  - this [notebook](notebooks/resampling_example.ipynb) illustrates how to use it
  - the strategy is implemented [here](./oversampling_strategies/)

If you want to reproduce our paper experiments:
  - the notebooks [here](notebooks/classif_experiments.ipynb) and [here](notebooks/distances_experiments.ipynb) reproduce the experiments
  - thise [code](./validation) contains implementation the protocols used for the numerical experiments of our article. 

## Data sets

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
* House_16H : https://www.openml.org/d/44123


An idea would be to add paper results:
Something like:

| Dataset    | Method 1 | Method 2 |
| -------- | ------- | ------- |
| [Pima](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  | xxx   | xxx   |
| [Phoneme](https://github.com/jbrownlee/Datasets/blob/master/phoneme.csv)  | xxx   | xxx   |
| [Abalone](https://archive.ics.uci.edu/dataset/1/abalone )   | xxx   | xxx   |

## Acknowledgements

This work was done through a partenership between **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.
