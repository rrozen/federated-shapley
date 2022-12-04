from typing import List, Callable

import numpy as np


class FederatedShapley:
    def __init__(self, data: List[np.ndarray], u: Callable[[np.ndarray], float], r: float) -> None:
        """
        :param data: List of datasets, one per participant
        :param u: valuation function
        :param r: max of u
        """
        self.data = data
        self.N = len(data)
        self.u = u
        self.r = r
        self.w = np.ones(10)
        self.rng = None

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
        # TODO
        return np.zeros_like(self.w)

    def roundSVEstimation(self, updates: List[np.ndarray], eps: float, delta: float) -> np.ndarray:
        """
        Computes Shapley values for participants at a given round
        :param updates: Weights updates for the round's participants
        :param eps: epsilon parameter
        :param delta: delta parameter
        :return: shapley values for the round
        """
        m = len(updates)
        A = int(2 * self.r ** 2 / eps ** 2 * np.log(2 * m / delta))
        s_hat = np.zeros(m)
        for a in range(A):
            uprev = self.u(self.w)
            permut = self.rng.permutation(updates)
            w = self.w
            for i, update in enumerate(permut):
                w = self.update_weights(w, update)
                unew = self.u(w)
                s_hat[i] += unew - uprev
                uprev = unew
        s_hat /= A
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

    def federatedSVEstimation(self, C: float, T: int, aggregation="sum", eps=0.1, delta=0.1, seed=None) -> np.ndarray:
        """
        Federated learning loop with Federated Shapley value computation
        :param C: Fraction of selected participants in each round
        :param T: Number of rounds
        :param aggregation: Weighting method
        :param eps: epsilon parameter
        :param delta: delta parameter
        :param seed: seed for the random numbers generator
        :return: Federated Data Shapley values
        """
        self.rng = np.random.default_rng(seed)
        S_hat = np.zeros(self.N)
        for t in range(T):
            m = int(C * self.N)
            participants = np.random.choice(self.N, size=m, replace=False)
            updates = [self.participant_update(k)
                       for k in participants]
            shapley_values = self.roundSVEstimation(updates, eps, delta)
            S_hat[participants] += self.weighted_shapley_values(shapley_values, t, aggregation)
            self.w = self.update_weights(self.w, np.mean(updates))
        return S_hat
