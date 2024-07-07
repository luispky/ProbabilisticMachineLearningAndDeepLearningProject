import numpy as np
import pandas as pd


def create_data(size, n_cols, high, k):
    mat = np.random.randint(0, high, (size, n_cols))

    columns = [f'x_{i}' for i in range(n_cols)]
    df = pd.DataFrame(mat, columns=columns)
    df['anomaly'] = np.sum(mat, axis=1) > k

    print(df.head(10))
    print(f'{np.average(df["anomaly"]):.1%} anomalies')

    df.to_csv('sum_limit_problem.csv', index=False)


if __name__ == '__main__':
    create_data(size=10_000, n_cols=5, high=4, k=10.5)
    # create_data(size=10_000, n_cols=4, high=2, k=3.5)
