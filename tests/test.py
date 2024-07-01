from src.datasets import SumCategoricalDataset, GaussianDataset


def test_1():
    dataset = SumCategoricalDataset(size=1000, n_values=(2, 3, 4), threshold=4.5)
    x = dataset.get_data()['x']
    y = dataset.get_data()['y']

    print(x[0])
    print(x.shape)
    print(y[0])
    print(y.shape)


def test_2():
    # todo: fix GaussianDataset
    dataset = GaussianDataset(size=1000, mean=0.5, cov=0.5)
    x = dataset.get_data()['x']
    y = dataset.get_data()['y']

    print(x[0])
    print(x.shape)
    print(y[0])
    print(y.shape)


if __name__ == '__main__':
    test_2()
