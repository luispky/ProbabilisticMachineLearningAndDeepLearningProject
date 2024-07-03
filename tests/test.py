import pandas as pd
from src.datasets import SumCategoricalDataset, GaussianDataset, DatabaseInterface


def test_1():
    """ author: Omar
    test SumCategoricalDataset
    """
    dataset = SumCategoricalDataset(size=1000, structure=(2, 3, 4), threshold=4.5)
    x = dataset.get_data()['x']
    y = dataset.get_data()['y']

    print(x[0])
    print(x.shape)
    print(y[0])
    print(y.shape)


def test_2():
    """
    # todo: fix GaussianDataset
    """
    dataset = GaussianDataset(size=1000, mean=0.5, cov=0.5)
    x = dataset.get_data()['x']
    y = dataset.get_data()['y']

    print(x[0])
    print(x.shape)
    print(y[0])
    print(y.shape)


def test_3(file_path='..\\datasets\\sample_data_preprocessed.csv'):
    """ author: Omar
    test DatabaseInterface
    """
    df = pd.read_csv(file_path)

    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # round Annual_Premium and Age
    df['Annual_Premium'] = df['Annual_Premium'].apply(lambda x: round(x))
    df['Age'] = df['Age'].apply(lambda x: round(x, -1))

    y = df.copy()['Response_1']

    del df['Response_1']

    data_x = DatabaseInterface(df)

    print('\n Column values:')
    for col in data_x.value_maps:
        print(f'{col:>25}  {len(data_x.value_maps[col])}  {data_x.inverse_value_maps[col]}')

    df_indices = data_x.convert_values_to_indices()

    print('\n')
    print(df_indices.head())
    print(y.head())


if __name__ == '__main__':
    test_3()
