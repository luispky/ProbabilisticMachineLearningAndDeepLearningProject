import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy


def generate_data(n, k):
    """
    Generate data for an n-bit anomaly detection problem
    The anomalous data-points have more than k ones in them
    """
    # list of all combination of n bits
    x = np.array([[i >> j & 1 for j in range(n)] for i in range(2**n)])

    # 1 if sum(x) > k else 0
    y = np.array([int(sum(v) > k) for v in x])
    y = y.reshape(-1, 1)

    return x, y


def training(n=8, k=4, n_epochs=3000):
    """ train a model on the problem"""
    x, y = generate_data(n, k)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    print(x.shape, y.shape)

    # model definition
    model = torch.nn.Sequential(
        torch.nn.Linear(n, 1),
        torch.nn.Sigmoid()
    )

    # loss function
    loss_fn = torch.nn.BCELoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

    # training loop
    for epoch in range(n_epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'\rEpoch {epoch}, Loss {loss.item():.3f}', end=' ')
    print()

    # test the model
    x_test, y_test = generate_data(n, k)
    y_pred = model(x_test)
    loss = loss_fn(y_pred, y_test)

    print(f'Loss on test data {loss.item():.3f}')

    # performance metrics
    y_class = (y_pred > 0.5).float()
    accuracy = (y_class == y_test).float().mean()
    print(f'Accuracy on test data {accuracy.item():.1%}')
    rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2))
    print(f'RMSE on test data {rmse.item():.3f}')

    # save the model
    name = f'model_{n}_{k}.pt'
    torch.save(model, name)


def reverse_classification(x, classifier, eta=0.1, n_iter=300, threshold=0.1, noise=0.01):
    """
    Given a classifier and a set of data points, modify the data points
    so that the classification changes from 1 to 0
    """
    x = deepcopy(x)

    # add gaussian noise to the input
    x += torch.randn_like(x) * noise

    x.requires_grad = True

    for i in range(n_iter):
        # Make the prediction
        y = classifier(x)

        # Compute the loss
        loss = torch.nn.BCELoss()(y, torch.zeros_like(y))

        # Compute the gradient of the loss with respect to x
        loss.backward()

        # Create a copy of x and update the copy
        x_copy = x.detach().clone()
        x_copy -= eta * x.grad

        # Update the original x with the modified copy
        x.data = x_copy

        # Clear the gradient for the next iteration
        x.grad.zero_()

        # print loss
        if loss.item() < threshold:
            print(f'Loss is {loss.item():.3f} < {threshold} at iteration {i}')
            break
        print(f'\rIteration {i+1}, Loss {loss.item():.3f}', end=' ')

    print()
    return x


def backward_gradient_method_test(n=8, k=4, n_example=1):
    """
    Apply gradient descent on the classification of the model
    in respect to the input x[i] so that the classification
    changes from 1 to 0
    """

    # generate data
    x, y = generate_data(n, k)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    model = torch.load(f'model_{n}_{k}.pt')

    # take an example
    y_positives = np.arange(len(x))[y.flatten() == 1]
    np.random.shuffle(y_positives)
    x_positives = x[y_positives]
    x_examples = x_positives[:n_example]

    # follow the gradient
    new_x = reverse_classification(x_examples, model)
    fig, ax = plt.subplots(2, 1)

    plt.sca(ax[0])
    plt.imshow(x_examples.detach().numpy(), cmap='gray')
    plt.clim(0, +1)

    plt.sca(ax[1])
    plt.imshow(new_x.detach().numpy(), cmap='gray')
    plt.clim(0, +1)

    print(x_examples.detach().numpy())
    print(np.round(new_x.detach().numpy(), 2))

    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    backward_gradient_method_test()
