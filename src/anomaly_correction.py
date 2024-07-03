import os
import numpy as np
import torch
from src.utils import cprint, bcolors, Probabilities
from src.inverse_gradient import InverseGradient
from src.datasets import SumCategoricalDataset


DEFAULT_MODEL_NAME = f'model_{os.path.splitext(os.path.basename(__file__))[0]}.pkl'


def main(n_data=1000, structure=(2, 3, 4), threshold=3.5, n_iter=100,
         critical_p=0.1, hidden=2, n_epochs=3000, lr=0.1,
         weight_decay=1e-4, momentum=0.9, n_data_examples=15,
         n_anomalies=3, model_name=DEFAULT_MODEL_NAME):
    """
    Example of how to run the inverse gradient
    algorithm on a synthetic dataset
    """
    np.random.seed(42)
    torch.set_default_dtype(torch.float64)

    # generate data
    dataset = SumCategoricalDataset(size=n_data, structure=structure, threshold=threshold)
    x = dataset.get_data()['x']
    y = dataset.get_data()['y']
    print(x.shape, y.shape)
    print(np.average(y.numpy()))

    # pick examples for display
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
    print(f'\n{len(y_positives)} positive examples')

    # plots: run the inverse gradient algorithm
    proba = Probabilities(structure)
    for row in range(n_anomalies):
        for col in range(n_anomalies):
            i = row * n_anomalies + col

            # pick an anomaly
            print(f'\n Anomaly {i+1})')
            anomaly = x_positives[i:i+1]

            # run the inverse gradient algorithm
            new_anomaly, loss = method.run(anomaly, structure, n_iter=n_iter, threshold=critical_p)
            corrected_anomaly = new_anomaly.detach().numpy()
            anomaly_probability = method.model(anomaly).item()
            new_anomaly_probability = method.model(new_anomaly).item()

            # print results
            x_ = anomaly.detach().numpy().astype(float)
            x_ = proba.prob_to_onehot(x_)
            v_ = proba.onehot_to_values(anomaly)
            x_new = corrected_anomaly.astype(float)
            v_new = proba.onehot_to_values(corrected_anomaly)
            changed = (np.sum(v_ != v_new, axis=1) > 0)[0]
            print(f'      x = {np.round(x_, 2)}')
            print(f'     x* = {np.round(x_new, 2)}')
            print(f'   dist = {np.linalg.norm(x_ - x_new):.3f}')
            print(f'      v = {v_}')
            print(f'     v* = {v_new}')
            cprint(f'changed = {changed}', bcolors.OKGREEN if changed else bcolors.FAIL)
            print(f'      p = {anomaly_probability:5.1%}')
            print(f'     p* = {new_anomaly_probability:5.1%}')


class AnomalyCorrection:
    """
    Final pipeline. Take as input a dataset, train a model, and
    perform anomaly correction.

    ================================================================

    Forward Steps:

    values -> indices + structure
    indices -> one-hot
    one-hot -> noisy_probabilities

    ================================================================

    Sideways Steps:

    noisy_probabilities + y (anomaly labels) -> classifier
    classifier + x (anomaly) -> corrected anomaly x* (probabilities)

    ================================================================

    Backward Steps:
    x* (probabilities) -> x* (one-hot)   # diffusion!
    x* (one-hot) -> x* (indices)
    x* (indices) -> x* (values)

    ================================================================
    """


