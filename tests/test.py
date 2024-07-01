import numpy as np
import pandas as pd
from src.datasets import SumCategoricalDataset, GaussianDataset, DatabaseInterface


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


def test_3():
    file_path = '..\\datasets\\sample_data_preprocessed.csv'
    df = pd.read_csv(file_path)

    # round Annual_Premium
    df['Annual_Premium'] = df['Annual_Premium'].apply(lambda x: round(x))

    # round Age to closest 10
    df['Age'] = df['Age'].apply(lambda x: round(x, -1))

    data = DatabaseInterface(df)

    print('\n Column values:')
    for col in data.inverse_value_maps:
        print(f'{col:>25}  {len(data.inverse_value_maps[col])}  {data.value_maps[col]}')

    df_indices = data.convert_to_value_indices()

    print('\n df after conversion:')
    print(df_indices.head())


if __name__ == '__main__':
    test_3()
