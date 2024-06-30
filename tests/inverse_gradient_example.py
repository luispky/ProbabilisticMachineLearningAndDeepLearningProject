import sys
import os
sys.path.append(os.path.abspath('..'))  # todo does this have to be here?
import numpy as np
import torch
from src.utils import cprint, bcolors, Probabilities
import matplotlib.pyplot as plt
from scripts.inverse_gradient import InverseGradient


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
    p = proba.add_noise(x)

    # convert to torch tensors
    p = torch.tensor(p, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)

    return p, y


class SumCapDataset:
    r""" Author: Omar
    Class to generate the dataset for the Inverse Gradient method
    """
    def __init__(self):
        self.dataset = None
        self.labels = None
        self.dataloader = None

    def generate_data(self, size, n_values: list | tuple, threshold):
        """
        Anomaly if sum of numbers is more than k
        """
        if self.dataset is not None:
            print('Data already generated')
            return self.dataset
        else:
            self.dataset = generate_data(size, n_values, threshold)

    def get_dataset_shape(self):
        assert self.dataset is not None, 'Dataset not generated'
        return self.dataset.shape

    def plot_data(self):
        raise NotImplementedError


def main(n_data=1000, n_values=(2, 2, 4), threshold=3.5, hidden=2,
         n_epochs=3000, lr=0.1,  weight_decay=1e-4, momentum=0.9,
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
            changed = (np.sum(v_ != v_new, axis=1) > 0)[0]
            print(f'      x = {np.round(x_, 2)}')
            print(f'     x* = {np.round(x_new, 2)}')
            print(f'   dist = {np.linalg.norm(x_ - x_new):.3f}')
            print(f'      v = {v_}')
            print(f'     v* = {v_new}')
            cprint(f'changed = {changed}', bcolors.OKGREEN if changed else bcolors.FAIL)
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
