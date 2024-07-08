import numpy as np
import pandas as pd


def sum_cap_dataset(size, n_cols, high, k, name='sum_cap_problem.csv'):
    """
    Anomaly if the sum of the row is greater than k
    """
    mat = np.random.randint(0, high, (size, n_cols))
    columns = [f'x_{i}' for i in range(n_cols)]
    df = pd.DataFrame(mat, columns=columns)
    df['anomaly'] = np.sum(mat, axis=1) > k

    print(df.head(10))
    print(f'{np.average(df["anomaly"]):.1%} anomalies')

    df.to_csv(name, index=False)


def no_repeat_dataset(size, n_cols, high, name='no_repeat_problem.csv'):
    """
    Anomaly if there are repeated values in the row
    """
    map = {i: chr(i + 65) for i in range(high)}
    mat = np.random.randint(0, high, (size, n_cols))
    columns = [f'x_{i}' for i in range(n_cols)]
    df = pd.DataFrame(mat, columns=columns)
    df = df.replace(map)
    df['anomaly'] = df.apply(lambda x: len(x) != len(set(x)), axis=1)

    print(df.head(10))
    print(f'{np.average(df["anomaly"]):.1%} anomalies')

    df.to_csv(name, index=False)


if __name__ == '__main__':
    # sum_cap_dataset(size=30_000, n_cols=5, high=4, k=10.5)
    # sum_cap_dataset(size=30_000, n_cols=4, high=2, k=3.5)
    no_repeat_dataset(size=30_000, n_cols=3, high=4)
