import numpy as np
import pandas as pd
from collections import Counter


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


def diff_divergence(p, q):
    """divergence between two probability distributions"""
    return np.linalg.norm(p - q, ord=1) / 2


def compute_divergence(a, b, divergence=diff_divergence):
    v_a = [str(a[i]) for i in range(a.shape[0])]
    v_b = [str(b[i]) for i in range(b.shape[0])]
    count_a = Counter(v_a)
    count_b = Counter(v_b)
    keys = set(count_a.keys()).union(set(count_b.keys()))
    p = np.array([count_a[key] for key in keys]) / len(a)
    q = np.array([count_b[key] for key in keys]) / len(b)
    return np.sum(divergence(p, q))


def test_divergence(size=10_000, n_cols=5, high=4):
    np.random.seed(42)
    a = np.random.randint(0, high, (size, n_cols))
    # b = np.random.randint(0, high, (size, n_cols))
    b = a
    print(f'\ndiv(a, b) = {compute_divergence(a, b):.3f}')


if __name__ == '__main__':
    # sum_cap_dataset(size=30_000, n_cols=5, high=4, k=10.5)
    # sum_cap_dataset(size=30_000, n_cols=4, high=2, k=3.5)
    # no_repeat_dataset(size=30_000, n_cols=3, high=4)
    test_divergence()
