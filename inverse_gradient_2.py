import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from misc import cprint, bcolors


def generate_data(size, n, r=1.):
    """anomaly if distance from origin > r"""
    x = np.random.normal(0, 1, (size, n))
    y = np.sum(x**2, axis=1) > r**2
    y = y.astype(np.float32).reshape(-1, 1)

    # convert to torch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x, y


class InverseGradient:

    def __init__(self, x: torch.tensor, y: torch.tensor, model_name):
        self.x = x
        self.y = y
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """"""
        name = self.get_model_name()
        try:
            cprint(f'Loading model from {name}', bcolors.WARNING)
            self.model = torch.load(name)
            cprint('Model loaded', bcolors.OKGREEN)
        except FileNotFoundError:
            cprint('Model not found', bcolors.FAIL)

    def get_model_name(self):
        return self.model_name

    def _training_loop(self, loss_fn, n_epochs, lr):

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # training loop
        for epoch in range(n_epochs):
            y_pred = self.model(self.x)
            loss = loss_fn(y_pred, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'\rlr={lr}, Epoch {epoch+1}, Loss {loss.item():.3f}', end=' ')
        print()

    def training(self, n_epochs=1000, lr=0.1):
        """ train a model on the problem"""
        assert self.model is not None

        # loss function
        loss_fn = torch.nn.BCELoss()

        # test the model
        y_pred = self.model(self.x)
        loss = loss_fn(y_pred, self.y)
        print(f'Initial Loss {loss.item():.3f}')

        # training
        self._training_loop(loss_fn, n_epochs, lr)
        self._training_loop(loss_fn, n_epochs, lr/10)

        # test the model
        y_pred = self.model(self.x)
        loss = loss_fn(y_pred, self.y)
        print(f'Final Loss {loss.item():.3f}')

        # performance metrics
        y_class = (y_pred > 0.5).float()
        accuracy = (y_class == self.y).float().mean()
        print(f'Accuracy on test data {accuracy.item():.1%}')
        rmse = torch.sqrt(torch.mean((y_pred - self.y) ** 2))
        print(f'RMSE on test data {rmse.item():.3f}')

        # save the model
        torch.save(self.model, self.get_model_name())

    def run(self, x, eta=0.01, n_iter=300, threshold=0.1):
        """
        Given a classifier and a set of data points, modify the data points
        so that the classification changes from 1 to 0
        """
        assert self.model is not None
        loss_value = 1
        x = deepcopy(x)

        # add gaussian noise to the input
        x.requires_grad = True

        for i in range(n_iter):
            # Make the prediction
            y = self.model(x)

            # Compute the loss
            loss = torch.nn.BCELoss()(y, torch.zeros_like(y))

            # Compute the gradient of the loss with respect to x
            loss.backward()

            # Create a copy of x and update the copy
            x_copy = x.detach().clone()
            dx = - eta * x.grad.sign()
            if i == 0:
                if np.linalg.norm(dx.numpy().flatten()) == 0:
                    cprint('Warning: Gradient is zero', bcolors.WARNING)
                    return x, loss_value
            x_copy += dx

            # Update the original x with the modified copy
            x.data = x_copy

            # Clear the gradient for the next iteration
            x.grad.zero_()

            # print loss
            loss_value = loss.item()
            if loss_value < threshold:
                cprint(f'Loss is {loss_value:.3f} < {threshold} at iteration {i}', bcolors.OKGREEN)
                return x, loss_value
            print(f'\rIteration {i+1}, Loss {loss_value:.3f}', end=' ')

        cprint('Warning: Maximum iterations reached', bcolors.WARNING)
        return x, loss_value


def main(size=200, n_dim=2, hidden=3, n_examples=20):
    np.random.seed(0)

    # generate numpy data
    x, y = generate_data(size, n_dim)
    print(x.shape, y.shape)

    # create inverse gradient
    inverse_grad = InverseGradient(x, y, model_name='model.pkl')

    # if model is not trained, train it
    if inverse_grad.model is None:

        # create the model
        cprint('Training model', bcolors.WARNING)
        inverse_grad.model = torch.nn.Sequential(
            torch.nn.Linear(n_dim, hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid()
        )

        # train the model
        inverse_grad.training(n_epochs=4000)
        cprint('Model trained', bcolors.OKGREEN)

        # save the model
        torch.save(inverse_grad.model, inverse_grad.get_model_name())
        cprint('Model saved', bcolors.OKGREEN)

    # pick an example of an anomaly (y==1)
    y_mask = y.flatten() == 1
    y_positives = np.arange(len(x))[y_mask]
    np.random.shuffle(y_positives)
    x_positives = x[y_positives]

    # plot
    fig, ax = plt.subplots(1, 2)

    plt.sca(ax[0])
    plt.title('Original data')
    plt.scatter(x[~y_mask][:, 0], x[~y_mask][:, 1], label='Normal')
    plt.scatter(x[y_mask][:, 0], x[y_mask][:, 1], label='Anomalies')
    plt.legend()

    plt.sca(ax[1])
    y_pred = inverse_grad.model(x).detach().numpy()

    plt.title('Trained model + Corrected anomalies')
    plt.scatter(x[:, 0], x[:, 1], label='Normal', s=10)
    plt.scatter(x[:, 0], x[:, 1], alpha=y_pred, label='Anomalies', s=10)
    plt.legend()

    # run the inverse gradient algorithm
    for i in range(n_examples):
        # pick an anomaly
        anomaly = x_positives[i:i+1]

        # run the inverse gradient algorithm
        new_anomaly, loss = inverse_grad.run(anomaly)
        corrected_anomaly = new_anomaly.detach().numpy()
        new_probability = inverse_grad.model(new_anomaly).item()

        # plot the corrected anomaly
        x0 = anomaly.numpy().flatten()
        x1 = corrected_anomaly.flatten()
        plt.plot(*zip(x0, x1), c='k', alpha=0.5)
        plt.scatter(*x0, c='red', label='Anomaly')
        plt.scatter(*x1, c='red', label='Corrected')
        plt.scatter(*x1, c='green', label='Corrected', alpha=1-new_probability)

    plt.show()


if __name__ == '__main__':
    main()
