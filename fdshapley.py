from itertools import permutations
from math import factorial
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from tqdm import tqdm

class FederatedShapley:
    def __init__(self, data_train: List[List[np.ndarray]], data_test: List[np.ndarray], clf_params: dict = {}) -> None:
        """
        :param data_train: Training data (data and labels) of all participants : [[x_1,y_1], ..., [x_N, y_N]]
        :param data_test:  Test data (data and labels): [x_test, y_test]
        :param clf_params: Parameters for the LogisticRegression
        """
        self.data_train = data_train
        self.data_test = data_test
        self.N = len(data_train)

        self.clf_params = clf_params
        self.clf_params.update({
                'max_iter': 1, 
                'warm_start': True,
                'fit_intercept':False
            })
        
        self.clf = LogisticRegression(**self.clf_params)
        self.clf.intercept_ = np.zeros(10)
        self.clf.classes_ = np.array(np.arange(10), dtype=str)

        _, d = self.data_train[0][0].shape

        self.w = np.random.rand(10, d) * 1e-3

        self.rng = None
        self.gamma = 1e-2

    def u(self, w: np.ndarray) -> float:
        """
        Compute the value of a set of weights (here, simply the accuracy)
        :param w: Weights
        :return: Score
        """
        self.clf.coef_ = w
        X_test, y_test = self.data_test
        return self.clf.score(X_test, y_test)

    @staticmethod
    def update_weights(w: np.ndarray, update: np.ndarray) -> np.ndarray:
        """
        Computes the updated weights.
        :param w: Old weights
        :param update: Weights (aggregated) update
        :return: New weights
        """
        return w + update

    def participant_update(self, k: int) -> np.ndarray:
        """
        Computes the weights update using participant k's data.
        :param k: ID of the participant
        :return: Weights update
        """
        clf = clone(self.clf)
        clf.coef_ = self.w.copy()
        xk, yk = self.data_train[k]
        clf.fit(xk, yk)
        update = clf.coef_ - self.w
        return update * self.gamma  # TODO

    def roundSVEstimation(self, updates: List[np.ndarray]) -> np.ndarray:
        """
        Computes Shapley values for participants at a given round
        :param updates: Weights updates for the round's participants
        :return: shapley values for the round
        """
        m = len(updates)
        s_hat = np.zeros(m)
        for permut in permutations(updates):
            uprev = self.u(self.w)
            w = self.w.copy()
            for i, update in enumerate(permut):
                w = self.update_weights(w, update)
                unew = self.u(w)
                s_hat[i] += unew - uprev
                uprev = unew
        s_hat /= factorial(len(updates))
        return s_hat

    @staticmethod
    def weighted_shapley_values(value: np.ndarray, t: int, aggregation: str):
        """
        Weight the Shapley value according to the round (maybe other parameters can be considered).
        :param value: Shapley values
        :param t: Round number
        :param aggregation: Weighting method
        :return: Weighted Shapley values
        """
        if aggregation == "sum":
            return value
        raise ValueError

    def federatedSVEstimation(self, C: float, T: int, aggregation="sum", seed=None) -> np.ndarray:
        """
        Federated learning loop with Federated Shapley value computation
        :param C: Fraction of selected participants in each round
        :param T: Number of rounds
        :param aggregation: Weighting method
        :param eps: epsilon parameter
        :param delta: delta parameters
        :param seed: seed for the random numbers generator
        :return: Federated Data Shapley values
        """
        self.rng = np.random.default_rng(seed)
        S_hat = np.zeros(self.N)
        #print("score", self.u(self.w))

        first_participation = [T]*self.N

        for t in tqdm(range(T)):   
            m = int(C * self.N)
            participants = np.random.choice(self.N, size=m, replace=False)
            for x in participants:
                if first_participation[x] == T:
                    first_participation[x] = t
            updates = [self.participant_update(k) for k in participants]
            shapley_values = self.roundSVEstimation(updates)
            S_hat[participants] += self.weighted_shapley_values(shapley_values, t, aggregation)
            self.w = self.update_weights(self.w, sum(updates))


            #print(f"{t=}") 
            #print(participants, shapley_values)
            #print("sum shapley", sum(shapley_values))
            #print("score", self.u(self.w))
            #print(S_hat)
          
        return S_hat, first_participation
