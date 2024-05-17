import random

import numpy as np
from imblearn.over_sampling import SMOTE


from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


class MySmote(BaseOverSampler): # to delete 
    """
    > Better naming of the class: what's the name in the paper ?
    > Should it inherit from imblearn's BaseOverSampler ?
    > It adds some work BUT it would be very nice for
        - users to be sure that they can trust the code
        - you to use all the checking methods that you can look at here
    > DOCstring: either Google or NumPy, took NumPy for example.

    > Here should describe class specificities briefly
    """

    def __init__(self, n_neighbors, w_simulation, w_simulation_params):
        """Instantiation of MySmote.


        Parameters
        ----------
        n_neighbors : type
            what it does

        w_simulation : type
            what is does

        w_simulation_params : type
            what is does
        """
        self.n_neighbors = n_neighbors
        self.w_simulation = w_simulation
        self.w_simulation_params = w_simulation_params

    def fit_resample(self, X, y=None, n_final_samples=None):
        """Resamples the dataset.


        if y=None, all points are considered positive, and oversampling on all X
        if n_final_sample=None, objective is balanced data.


        Parameters
        ----------
        X : type
            what it does

        y : type
            what is does, default is None.

        n_final_samples : type
            what is does, default is None.

        Returns
        -------
        oversampled_X : type
            what it is
        oversampled_y : type
            what it is
        """
        if n_final_samples is None and y is None:
            raise ValueError(
                "You need to provide a value for n_final_samples or y, got None and None."
            )

        if y is None:
            majority_X = X  # English
            minority_X = np.ones((0, X.shape[1]))
        else:
            majority_X = X[y == 1]
            minority_X = X[y == 0]
            if n_final_samples is None:
                n_final_samples = (y == 0).sum()

        majority_n = majority_X.shape[0]  # If 1 MUST be majority, need to be precised # Unsused ?
        # Same in ImbLearn ?
        # Why not take two cases in consideration: np.max([np.sum(y == 1), np.sum(y == 0)]) ?

        nearest_neighbor = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm="ball_tree"
        )
        nearest_neighbor.fit(majority_X)
        neighbor_by_index = nearest_neighbor.kneighbors(
            X=majority_X, n_neighbors=self.n_neighbors + 1, return_distance=False
        )
        # the first element is always the given point (my nearest neighbor is myself).

        minority_n = minority_X.shape[0] # ?
        n_synthetic_samples = n_final_samples - minority_n
        new_samples = np.zeros((n_synthetic_samples, X.shape[1]))
        for i in range(n_synthetic_samples):
            central_index = np.random.randint(
                minority_n
            )  # central point that is chosen

            chosen_neighbor = np.random.randint(
                1, self.n_neighbors + 1
            )  # nearest neighbors which is choisen. The central point itself is excluded (0 excluded)
            neighbor_index = neighbor_by_index[central_index][chosen_neighbor]

            weight = self.w_simulation(
                **self.w_simulation_params
            )  # factor of the difference betweeen the central point and the selected neighbor
            new_samples[i, :] = majority_X[central_index] + weight * (
                majority_X[neighbor_index] - majority_X[central_index]
            )

        oversampled_X = np.concatenate((minority_X, majority_X, new_samples), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(minority_X), 0), np.full((n_final_samples,), 1))
        )

        return oversampled_X, oversampled_y


class CVSmoteModel(object):
    """
    CVSmoteModel. It's an estimator and not a oversampling strategy only like the others class in this file.
    """

    def __init__(self, splitter, model, list_k_max=100, list_k_step=10):
        """_summary_

        Parameters
        ----------
        splitter : sk-learn spliter object (or child)
            _description_
        model : _type_
            _description_
        list_k_max : int, optional
            _description_, by default 100
        list_k_step : int, optional
            _description_, by default 10
        """
        self.splitter = splitter
        self.list_k_max = list_k_max  # why is it called list ?
        self.list_k_step = list_k_step  # why is it called list ?
        self.model = model
        self.estimators_ = [0]  # are you sure about it ?

    def fit(self, X, y, sample_weight=None):
        """
        X and y are numpy arrays
        sample_weight is a numpy array
        """

        n_positifs = np.array(y, dtype=bool).sum()
        list_k_neighbors = [
            5,
            max(int(0.01 * n_positifs), 1),
            max(int(0.1 * n_positifs), 1),
            max(int(np.sqrt(n_positifs)), 1),
        ]
        list_k_neighbors.extend(
            list(np.arange(1, self.list_k_max, self.list_k_step, dtype=int))
        )

        best_score = -1
        folds = list(
            self.splitter.split(X, y)
        )  # you really need to transform it into a list ?
        for k in list_k_neighbors:
            scores = []
            for train, test in folds:
                new_X, new_y = SMOTE(k_neighbors=k).fit_resample(X[train], y[train])
                self.model.fit(new_X, new_y, sample_weight)
                scores.append(
                    roc_auc_score(y[test], self.model.predict_proba(X[test])[:, 1])
                )
            if sum(scores) > best_score:
                best_k = k

        new_X, new_y = SMOTE(k_neighbors=best_k).fit_resample(X, y)
        self.model.fit(new_X, new_y, sample_weight)
        if hasattr(self.model, "estimators_"):
            self.estimators_ = self.model.estimators_

    def predict(self, X):
        """
        predict
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        predict_probas
        """
        return self.model.predict_proba(X)


class MGS(BaseOverSampler):
    """
    MGS
    """

    def __init__(
        self, K, n_points, llambda, sampling_strategy="auto", random_state=None
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        self.n_points = n_points
        self.random_state = random_state  # Add ?

    def _fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        if n_final_sample=None, objective is balanced data.
        """

        if y is None:
            X_positifs = X
            X_negatifs = np.ones((0, X.shape[1]))
            assert (
                n_final_sample is not None
            ), "You need to provide a number of final samples."
        else:
            X_positifs = X[y == 1]
            X_negatifs = X[y == 0]
            if n_final_sample is None:
                n_final_sample = (y == 0).sum()

        n_minoritaire = X_positifs.shape[0]
        dimension = X.shape[1]
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbor_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=False
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension))
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire)
            indices_neigh = [
                0
            ]  ## the central point is selected for the expectation and covariance matrix
            indices_neigh.extend(
                random.sample(range(1, self.K + 1), self.n_points)
            )  # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[indice][indices_neigh]
            mu = (1 / self.n_points) * X_positifs[indice_neighbors, :].sum(axis=0)
            sigma = (
                self.llambda
                * (1 / self.n_points)
                * (X_positifs[indice_neighbors, :] - mu).T.dot(
                    (X_positifs[indice_neighbors, :] - mu)
                )
            )

            new_observation = np.random.multivariate_normal(
                mu, sigma, check_valid="raise"
            ).T
            new_samples[i, :] = new_observation

        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y


class MGS2(BaseOverSampler):
    """
    MGS
    """

    def __init__(
        self, K, llambda, sampling_strategy="auto", random_state=None
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        self.random_state = random_state  # Add ?

    def _fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        if n_final_sample=None, objective is balanced data.
        """

        if y is None:
            X_positifs = X
            X_negatifs = np.ones((0, X.shape[1]))
            assert (
                n_final_sample is not None
            ), "You need to provide a number of final samples."
        else:
            X_positifs = X[y == 1]
            X_negatifs = X[y == 0]
            if n_final_sample is None:
                n_final_sample = (y == 0).sum()

        n_minoritaire = X_positifs.shape[0]
        dimension = X.shape[1]
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbors_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=False
        )

        n_synthetic_sample = n_final_sample - n_minoritaire


        # computing mu and covariance at once for every minority class points
        all_neighbors = X_positifs[neighbors_by_index.flatten()]
        mus = (1 / self.K) * all_neighbors.reshape(len(X_positifs), self.K + 1, dimension).sum(axis=1)
        centered_X = X_positifs[neighbors_by_index.flatten()] - np.repeat(mus, self.K + 1, axis=0)
        centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension)
        covs = self.llambda * np.matmul(np.swapaxes(centered_X,1,2), centered_X) / self.K
        
        # spectral decomposition of all covariances
        eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
        eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
        Diag = (np.eye(eigen_values.shape[1]) * eigen_values[:, np.newaxis])
        As = np.matmul(eigen_vectors,Diag)

        # sampling all new points
        #u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        #new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension))
        for i,central_point in enumerate(indices):
            u = np.random.normal(loc=0, scale=1, size=dimension)
            new_observation = mus[central_point, :] + As[central_point].dot(u)
            new_samples[i, :] = new_observation
        
        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )
        
        return oversampled_X, oversampled_y


class NoSampling(object):
    """
    None rebalancing strategy class
    """

    def fit_resample(self, X, y):
        """
        X is a numpy array
        y is a numpy array of dimension (n,)
        """
        return X, y


