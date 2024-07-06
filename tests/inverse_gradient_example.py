import sys
import os
sys.path.append(os.path.abspath('..'))  # todo does this have to be here?
import numpy as np
import torch
from src.utils import cprint, bcolors, Probabilities
import matplotlib.pyplot as plt
from src.old_inverse_gradient import InverseGradient
from src.datasets import SumCategoricalDataset


DEFAULT_MODEL_NAME = f'../models/inverse_gradient_model.pkl'


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
    fig, ax = plt.subplots(n_anomalies, n_anomalies)
    plt.suptitle('Examples of Inverse Gradient Algorithm\nfor anomaly correction')
    proba = Probabilities(structure)
    for row in range(n_anomalies):
        for col in range(n_anomalies):
            plt.sca(ax[row, col])
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

            # plot
            plt.bar(range(len(x_[0])), x_[0], color='w', edgecolor='k', label='Original')
            plt.bar(range(len(x_new[0])), x_new[0], alpha=0.5, label='Corrected')

            sum_ = 0
            for j_ in range(len(structure)):
                # plot rectangle
                x0 = sum_ + v_new[0][j_]
                y0 = x_new[0][sum_ + v_new[0][j_]] / 2
                plt.scatter([x0], [y0], c='k', alpha=0.15)
                plt.gca().add_patch(plt.Rectangle((sum_-0.5, -0.02), structure[j_], 1.04,
                                                  linewidth=2,
                                                  edgecolor='lightgray', facecolor='none'))
                sum_ += structure[j_]

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
