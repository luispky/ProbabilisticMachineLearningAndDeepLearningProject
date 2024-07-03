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
    This method is used to correct anomalies in a dataset
    by using the inverse gradient algorithm.
    """
    def __init__(self, x: torch.tensor, y: torch.tensor, model_name: str):
        self.x = x
        self.y = y
        self.model_name = model_name
        self.model = None
        self._p_copy = None
        self._module = None
        self._load_model()

    def _load_model(self):
        """Load a model from a file"""
        name = self.get_model_name()
        try:
            cprint(f'Loading model from {name}', bcolors.WARNING)
            self.model = torch.load(name)
            cprint('Model loaded', bcolors.OKGREEN)
        except FileNotFoundError:
            cprint('Model not found', bcolors.FAIL)

    def get_model_name(self):
        """Get the model name"""
        return self.model_name

    def _training_loop(self, loss_fn, n_epochs, lr=0.1, weight_decay=1e-3, momentum=0.9, nesterov=True):
        """training loop"""
        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay, momentum=momentum,
                                    nesterov=nesterov)

        # training loop
        for epoch in range(n_epochs):
            y_pred = self.model(self.x)
            loss = loss_fn(y_pred, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'\rlr={lr}, Epoch {epoch+1}, Loss {loss.item():.6f}', end=' ')
        print()

    def training(self, n_epochs=1000, lr=0.1, weight_decay=1e-3, momentum=0.9, nesterov=True):
        """ train a model on the problem"""
        assert self.model is not None

        # loss function
        loss_fn = torch.nn.BCELoss()

        # test the model
        y_pred = self.model(self.x)
        loss = loss_fn(y_pred, self.y)
        print(f'Initial Loss {loss.item():.3f}')

        # training
        self._training_loop(loss_fn, n_epochs, lr=lr, weight_decay=weight_decay,
                            momentum=momentum, nesterov=nesterov)
        self._training_loop(loss_fn, n_epochs, lr=lr/10, weight_decay=weight_decay,
                            momentum=momentum, nesterov=nesterov)

        # test the model
        y_pred = self.model(self.x)
        loss = loss_fn(y_pred, self.y)
        print(f'Final Loss {loss.item():.6f}')

        # performance metrics
        y_class = (y_pred > 0.5).float()
        accuracy = (y_class == self.y).float().mean()
        dummy_acc = max(self.y.mean().item(), 1 - self.y.mean().item())
        acc = accuracy.item()
        usefulness = max([0, (acc - dummy_acc) / (1 - dummy_acc)])
        if usefulness > 0.75:
            color = bcolors.OKGREEN
        elif usefulness > 0.25:
            color = bcolors.WARNING
        else:
            color = bcolors.FAIL
        print(f'Dummy accuracy = {dummy_acc:.1%}')
        print(f'Accuracy on test data = {acc:.1%}')
        cprint(f'usefulness = {usefulness:.1%}', color)
        rmse = torch.sqrt(torch.mean((y_pred - self.y) ** 2))
        print(f'RMSE on test data {rmse.item():.3f}')

        # save the model
        torch.save(self.model, self.get_model_name())

    def _compute_p_copy(self, p, proba, eta):
        self._p_copy = p.detach().clone()
        dp = - p.grad
        self._module = np.linalg.norm(dp.numpy().flatten())
        dp = dp / self._module * eta
        self._p_copy += dp
        self._p_copy = proba.normalize(self._p_copy)

    def run(self, x, n_values, eta=0.01, n_iter=100, threshold=0.1):
        """
        Given a classifier and a set of data points, modify the data points
        so that the classification changes from 1 to 0.

        params:
        x: torch.tensor, the data point to modify
        n_values: tuple, the number of values for each feature
        eta: float, the step size for the gradient descent
        n_iter: int, the maximum number of iterations
        threshold: float, the threshold probability for the loss function
        """
        assert self.model is not None
        assert x.shape[0] == 1
        assert 0 < eta < 1
        assert 0 < threshold < 1

        p = deepcopy(x)
        proba = Probabilities(n_values)
        v_old = proba.onehot_to_values(p)[0]

        # add gaussian noise to the input
        p.requires_grad = True

        i = 0
        while True:
            i += 1

            # Make the prediction
            y = self.model(p)

            # Compute the loss
            loss = torch.nn.BCELoss()(y, torch.zeros_like(y))

            # Compute the gradient of the loss with respect to x
            loss.backward()

            # Create a copy of x and update the copy
            self._compute_p_copy(p, proba, eta)

            # Update the original x with the modified copy
            p.data = self._p_copy

            # Clear the gradient for the next iteration
            p.grad.zero_()

            # Check if v_new is different to v_old
            v_new = proba.onehot_to_values(p.detach().numpy())[0]
            changed = np.any(v_old != v_new)

            # check if the loss is below the threshold
            loss_value = loss.item()
            print(f'\rIteration {i+1}, Loss {loss_value:.3f}', end=' ')
            if loss_value < threshold:
                if changed:
                    cprint(f'\rIteration {i}) loss is {loss_value:.3f} < {threshold}', bcolors.OKGREEN)
                    break

            # check if the gradient is zero
            if self._module == 0:
                cprint('Warning: Gradient is zero', bcolors.WARNING)
                break

            # check if the maximum number of iterations is reached
            if i == n_iter:
                cprint('Warning: Maximum iterations reached', bcolors.WARNING)
                break

        return p, loss_value
