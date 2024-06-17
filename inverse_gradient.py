import numpy as np
import torch


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


def backward_gradient_method(n=8, k=4):
    """
    Apply gradient descent on the classification of the model
    in respect to the input x[i] so that the classification
    changes from 1 to 0
    """
    # generate data
    x, y = generate_data(n, k)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    model = torch.load('model_8_4.pt')

    # take an example
    zeros = np.array(y == 1).reshape(-1)
    indices = np.arange(len(y))[zeros]
    i = np.random.choice(indices)
    example = {"x": x[i], "y": y[i]}
    print(example)

    # follow the gradient
    for i in range(1000):
        # modify the example slightly so that it changes the classification
        x_var = torch.autograd.Variable(example["x"], requires_grad=True)
        y_var = model(x_var)
        loss = (1 - y_var) ** 2
        loss.backward()
        example["x"] = x_var.data
        print(f'\rIteration {i}, Loss {loss.item():.3f}', end=' ')

        # ok but how tho...?


if __name__ == '__main__':
    backward_gradient_method()
