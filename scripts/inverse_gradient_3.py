import numpy as np
import torch
from copy import deepcopy
from misc import cprint, bcolors


class InverseGradient:

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
        self._training_loop(loss_fn, n_epochs, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        self._training_loop(loss_fn, n_epochs, lr=lr/10, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

        # test the model
        y_pred = self.model(self.x)
        loss = loss_fn(y_pred, self.y)
        print(f'Final Loss {loss.item():.6f}')

        # performance metrics
        y_class = (y_pred > 0.5).float()
        accuracy = (y_class == self.y).float().mean()
        print(f'Accuracy on test data {accuracy.item():.1%}')
        rmse = torch.sqrt(torch.mean((y_pred - self.y) ** 2))
        print(f'RMSE on test data {rmse.item():.3f}')

        # save the model
        torch.save(self.model, self.get_model_name())

    # def _update_model(self):

    def run(self, x, eta=0.01, n_iter=300, threshold=0.1):
        """
        Given a classifier and a set of data points, modify the data points
        so that the classification changes from 1 to 0
        """
        assert self.model is not None
        x = deepcopy(x)

        # add gaussian noise to the input
        x.requires_grad = True

        i = 0
        while True:
            i += 1

            # Make the prediction
            y = self.model(x)

            # Compute the loss
            loss = torch.nn.BCELoss()(y, torch.zeros_like(y))

            # Compute the gradient of the loss with respect to x
            loss.backward()

            # Create a copy of x and update the copy
            x_copy = x.detach().clone()
            dx = - x.grad
            module = np.linalg.norm(dx.numpy().flatten())

            dx = dx / module * eta
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

            if module == 0:
                cprint('Warning: Gradient is zero', bcolors.WARNING)
                return x, loss_value

            if i == n_iter:
                cprint('Warning: Maximum iterations reached', bcolors.WARNING)
                return x, loss_value


def generate_data(size, n, m, k):
    """anomaly if sum of numbers is more than k"""
    # raw data
    x = np.random.randint(0, m, size=(size, n))
    y = np.sum(x, axis=1) > k

    # convert to one-hot encoding
    x1 = np.zeros((size, n * m), dtype=np.float64)
    for i in range(size):
        for j in range(n):
            x1[i, x[i, j] + j * m] = 1
    y = np.reshape(y, (size, 1))

    # convert to torch tensors
    x = torch.tensor(x1, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    return x, y


def update_probability(p, dp):  # todo replace with fancy version
    """Update probability"""
    q = p + dp
    q = np.maximum(0, q)
    return q / np.sum(q)


def main(n_data=10_000, n_numbers=2, n_values=2, threshold=1.5,
         hidden=2, n_epochs=1000, lr=0.1, weight_decay=1e-3, momentum=0.9,
         n_data_examples=15, n_anomalies=2):
    np.random.seed(42)
    torch.set_default_dtype(torch.float64)

    # generate data
    x, y = generate_data(n_data, n_numbers, n_values, threshold)
    print(x.shape, y.shape)

    # examples
    for i in range(n_data_examples):
        print(x[i].detach().numpy().astype(int), y[i].detach().numpy().astype(int))

    print(f'\n{np.average(y.numpy()):.1%} of data is positive')

    method = InverseGradient(x, y, 'model_3.pkl')

    # if model is not trained, train it
    if method.model is None:
        input_size = x.shape[1]
        print(f'Input size: {input_size}')

        # create the model
        cprint('Training model', bcolors.WARNING)
        method.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden, hidden),
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
    for i in range(n_anomalies):
        # pick an anomaly
        print(f'\n Anomaly {i})')
        anomaly = x_positives[i:i+1]

        # run the inverse gradient algorithm
        new_anomaly, loss = method.run(anomaly)
        corrected_anomaly = new_anomaly.detach().numpy()
        new_anomaly_probability = method.model(new_anomaly).item()

        # print results
        print(f' x = {anomaly.detach().numpy().astype(float)}')
        print(f'x* = {np.round(corrected_anomaly.astype(float), 2)}')
        print(f'p* = {new_anomaly_probability:.3f}')


if __name__ == '__main__':
    main()
