""" Author: Omar
In this module, we use the inverse gradient algorithm to correct anomalies in a dataset.
The algorithm is used to correct anomalies in a dataset by using the inverse gradient algorithm.
"""
import numpy as np
import torch
from copy import deepcopy
from src.utils import cprint, bcolors, Probabilities
import os
import matplotlib.pyplot as plt


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
        self._training_loop(loss_fn, n_epochs, lr=lr,
                            weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        self._training_loop(loss_fn, n_epochs, lr=lr/10,
                            weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

        # test the model
        y_pred = self.model(self.x)
        loss = loss_fn(y_pred, self.y)
        print(f'Final Loss {loss.item():.6f}')

        # performance metrics
        y_class = (y_pred > 0.5).float()
        accuracy = (y_class == self.y).float().mean()
        if accuracy.item() > 0.95:
            color = bcolors.OKGREEN
        elif accuracy.item() > 0.8:
            color = bcolors.WARNING
        else:
            color = bcolors.FAIL
        cprint(f'Accuracy on test data {accuracy.item():.1%}', color)
        rmse = torch.sqrt(torch.mean((y_pred - self.y) ** 2))
        print(f'RMSE on test data {rmse.item():.3f}')

        # save the model
        torch.save(self.model, self.get_model_name())

    # def _update_model(self):

    def run(self, x, n_values, eta=0.01, n_iter=300, threshold=0.05):
        """
        Given a classifier and a set of data points, modify the data points
        so that the classification changes from 1 to 0
        """
        assert self.model is not None
        p = deepcopy(x)
        proba = Probabilities(n_values)

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
            p_copy = p.detach().clone()
            dp = - p.grad
            module = np.linalg.norm(dp.numpy().flatten())
            dp = dp / module * eta
            p_copy += dp
            p_copy = proba.normalize(p_copy)

            # Update the original x with the modified copy
            p.data = p_copy

            # Clear the gradient for the next iteration
            p.grad.zero_()

            # print loss
            loss_value = loss.item()
            if loss_value < threshold:
                cprint(f'\rIteration {i}) loss is {loss_value:.3f} < {threshold}', bcolors.OKGREEN)
                return p, loss_value
            print(f'\rIteration {i+1}, Loss {loss_value:.3f}', end=' ')

            if module == 0:
                cprint('Warning: Gradient is zero', bcolors.WARNING)
                return p, loss_value

            if i == n_iter:
                cprint('Warning: Maximum iterations reached', bcolors.WARNING)
                return p, loss_value


def generate_data(size, n_values: list | tuple, threshold):
    """
    Anomaly if sum of numbers is more than k
    """
    proba = Probabilities(n_values)

    # raw data
    p = np.random.random(size=(size, sum(n_values)))
    p = proba.normalize(p)

    x = proba.prob_to_onehot(p)
    values = proba.onehot_to_values(x)
    y = np.sum(values, axis=1) > threshold
    y = np.expand_dims(y, axis=1)

    # convert to torch tensors
    x = torch.tensor(p, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)

    return x, y


def update_probability(p, dp):  # todo replace with fancy version
    """Update probability"""
    q = p + dp
    q = np.maximum(0, q)
    return q / np.sum(q)


def main(n_data=1000, n_values=(2, 2, 4), threshold=3.5, hidden=2,
         n_epochs=2000, lr=0.1,  weight_decay=1e-3, momentum=0.9,
         n_data_examples=15, n_anomalies=3, model_name=None):
    np.random.seed(42)
    torch.set_default_dtype(torch.float64)

    # set model name
    if model_name is None:
        model_name = f'model_{os.path.splitext(os.path.basename(__file__))[0]}.pkl'

    # generate data
    x, y = generate_data(n_data, n_values, threshold)
    print(x.shape, y.shape)
    print(np.average(y.numpy()))

    # examples
    for i in range(n_data_examples):
        x_ = x[i].detach().numpy().astype(float)
        y_ = y[i].detach().numpy().astype(float)
        print(np.round(x_, 2), np.round(y_, 1))

    print(f'\n{np.average(y.numpy()):.1%} of data is positive')

    method = InverseGradient(x, y, model_name)

    # if model is not trained, train it
    if method.model is None:
        input_size = x.shape[1]
        print(f'Input size: {input_size}')

        # create the model
        cprint('Training model', bcolors.WARNING)
        method.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid()
        )

        # train the model
        method.training(n_epochs=n_epochs, lr=lr, weight_decay=weight_decay, momentum=momentum)
        cprint('Model trained', bcolors.OKGREEN)

        # save the model
        torch.save(method.model, method.get_model_name())
        cprint('Model saved', bcolors.OKGREEN)

    # pick examples of an anomaly (y==1)
    y_mask = y.flatten() == 1
    y_positives = np.arange(len(x))[y_mask]
    np.random.shuffle(y_positives)
    x_positives = x[y_positives]

    # positives
    print(f'\n{len(y_positives)} positive examples')

    # run the inverse gradient algorithm
    fig, ax = plt.subplots(n_anomalies, n_anomalies)
    plt.suptitle('Examples of Inverse Gradient Algorithm\nfor anomaly correction')
    proba = Probabilities(n_values)
    for row in range(n_anomalies):
        for col in range(n_anomalies):
            plt.sca(ax[row, col])
            i = row * n_anomalies + col

            # pick an anomaly
            print(f'\n Anomaly {i+1})')
            anomaly = x_positives[i:i+1]

            # run the inverse gradient algorithm
            new_anomaly, loss = method.run(anomaly, n_values, threshold=0.01)
            corrected_anomaly = new_anomaly.detach().numpy()
            anomaly_probability = method.model(anomaly).item()
            new_anomaly_probability = method.model(new_anomaly).item()

            # print results
            x_ = anomaly.detach().numpy().astype(float)
            x_ = proba.prob_to_onehot(x_)
            v_ = proba.onehot_to_values(anomaly)
            x_new = corrected_anomaly.astype(float)
            v_new = proba.onehot_to_values(corrected_anomaly)
            print(f'      x = {np.round(x_, 2)}')
            print(f'     x* = {np.round(x_new, 2)}')
            print(f'   dist = {np.linalg.norm(x_ - x_new):.3f}')
            print(f'      v = {v_}')
            print(f'     v* = {v_new}')
            print(f'changed = {np.sum(v_ != v_new, axis=1) > 0}')
            print(f'      p = {anomaly_probability:5.1%}')
            print(f'     p* = {new_anomaly_probability:5.1%}')

            # plot
            plt.bar(range(len(x_[0])), x_[0], color='w', edgecolor='k', label='Original')
            plt.bar(range(len(x_new[0])), x_new[0], alpha=0.5, label='Corrected')

            sum_ = 0
            for j_ in range(len(n_values)):
                # plot rectangle
                x0 = sum_ + v_new[0][j_]
                y0 = x_new[0][sum_ + v_new[0][j_]] / 2
                plt.scatter([x0], [y0], c='k', alpha=0.15)
                plt.gca().add_patch(plt.Rectangle((sum_-0.5, -0.02), n_values[j_], 1.04,
                                                  linewidth=2,
                                                  edgecolor='lightgray', facecolor='none'))
                sum_ += n_values[j_]

            if row + col == 0:
                plt.legend()
                plt.xlabel('Features')
                plt.ylabel('Probability')
            else:
                plt.yticks([])
            plt.xticks([])

    plt.show()


if __name__ == '__main__':
    main()
