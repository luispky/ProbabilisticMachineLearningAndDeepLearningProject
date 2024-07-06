""" Author: Omar
In this module, we use the inverse gradient algorithm to correct anomalies in a dataset.
The algorithm is used to correct anomalies in a dataset by using the inverse gradient algorithm.
"""
import numpy as np
import torch
from copy import deepcopy
from src.utils import cprint, bcolors, Probabilities


class InverseGradient:
    """ Author: Omar
    Bare-bones inverse gradient method
    """
    def __init__(self, model):
        self.model = model
        self._p_copy = None
        self._module = None

    def _compute_p_copy(self, p, proba, eta):
        """ Update probabilities by maintaining normalization """
        self._p_copy = p.detach().clone()
        dp = -p.grad
        self._module = np.linalg.norm(dp.numpy().flatten())
        if self._module == 0:
            raise ZeroDivisionError
        dp = dp / self._module * eta
        self._p_copy += dp
        self._p_copy = proba.normalize(self._p_copy)

    def run(self, p: torch.tensor, structure: tuple, eta=0.01, n_iter=100, threshold_p=0.1):
        """
        Given a classifier and a set of data points, modify the data points
        so that the classification changes from 1 to 0.

        params:
        p: torch.tensor, the data point to modify
        structure: tuple, the number of values for each feature
        eta: float, the step size for the gradient descent
        n_iter: int, the maximum number of iterations
        threshold: float, the threshold probability for the loss function
        """
        assert self.model is not None
        assert p.shape[0] == 1


        assert 0 < eta < 1
        assert 0 < threshold_p < 1

        p_ = deepcopy(p)
        proba = Probabilities(structure)
        v_old = proba.onehot_to_values(p_)[0]
        v_new = v_old.copy()
        mask = v_new == v_new
        success = True

        # add gaussian noise to the input
        p_.requires_grad = True

        i = 0
        while True:
            i += 1

            # Make the prediction
            y = self.model(p_)
            p_anomaly = y[0][0]

            # Compute the loss
            loss = torch.nn.BCELoss()(y, torch.zeros_like(y))

            # Compute the gradient of the loss with respect to x
            loss.backward()

            # Create a copy of x and update the copy
            try:
                self._compute_p_copy(p_, proba, eta)
            except ZeroDivisionError:
                # check if the gradient is zero
                cprint('Warning: Gradient is zero', bcolors.WARNING)
                success = False
                break

            # Update the original x with the modified copy
            p_.data = self._p_copy

            # Clear the gradient for the next iteration
            p_.grad.zero_()

            # Check if v_new is different to v_old
            v_new = proba.onehot_to_values(p_.detach().numpy())[0]
            mask = v_old != v_new
            changed = np.any(mask)

            # check if the loss is below the threshold
            print(f'\rIteration {i+1}, pred {p_anomaly:.1%}', end=' ')
            if p_anomaly < threshold_p:
                if changed:
                    cprint(f'\rIteration {i}) loss is {p_anomaly:.1%} < {threshold_p:.1%}', bcolors.OKGREEN)
                    break

            # check if the maximum number of iterations is reached
            if i == n_iter:
                cprint(f'Warning: Maximum iterations reached ({p_anomaly:.1%})', bcolors.WARNING)
                success = False
                break

        return {"values": v_new, "mask": mask, "proba": p_, "anomaly_p": p_anomaly, "success": success}
