import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cprint(text, color, end='\n'):
    print(color + text + bcolors.ENDC, end=end)


class Probabilities:
    """This class helps normalize probabilities for a set of features with different number of values"""
    def __init__(self, n_values: list):
        self.n_values = n_values
        self.n = len(n_values)
        self.length = sum(n_values)
        self.mat = None
        self._set_mat()

    def _set_mat(self):
        """Create binary masks that divide the various features"""
        self.mat = np.zeros((self.length, self.length), dtype=np.float64)
        for i in range(self.n):
            start = sum(self.n_values[:i])
            for j in range(self.n_values[i]):
                self.mat[start:start+j+1, start:start+self.n_values[i]] = 1

    def normalize(self, p):
        """Cap at 0, then normalize the probabilities for each feature"""
        assert len(p.shape) == 2, f'{len(p.shape)} != 2'
        assert p.shape[1] == self.length, f'{p.shape[1]} != {self.length}'
        p = np.maximum(0, p)
        s = np.dot(p, self.mat)
        assert np.all(s > 0), f'Zero sum: {s}'
        return p / s

    def to_onehot(self, x):
        """Convert the original values to one-hot encoding"""
        assert len(x.shape) == 2, f'{len(x.shape)} != 2'
        assert x.shape[1] == self.n, f'{x.shape[1]} != {self.n}'
        # check that each value of x is less than the number of values for that feature
        assert np.all(np.max(x, axis=0) < self.n_values), f'Values out of range'
        # check that values are positive
        assert np.all(x >= 0), f'Negative values'

        x1 = np.zeros((x.shape[0], self.length), dtype=np.float64)
        start = 0
        for i in range(self.n):
            x1[np.arange(x.shape[0]), x[:, i] + start] = 1
            start += self.n_values[i]
        return x1

    def onehot_to_values(self, x):
        """Return the original values from the one-hot encoding"""
        assert len(x.shape) == 2, f'{len(x.shape)} != 2'
        assert x.shape[1] == self.length, f'{x.shape[1]} != {self.length}'
        x1 = np.zeros((x.shape[0], self.n), dtype=np.int64)
        start = 0
        for i in range(self.n):
            x1[:, i] = np.argmax(x[:, start:start+self.n_values[i]], axis=1)
            start += self.n_values[i]
        return x1

    def prob_to_onehot(self, p):
        """Convert the probabilities to one-hot encoding"""
        assert len(p.shape) == 2, f'{len(p.shape)} != 2'
        assert p.shape[1] == self.length, f'{p.shape[1]} != {self.length}'
        x = np.zeros((p.shape[0], self.n), dtype=np.int64)
        start = 0
        for i in range(self.n):
            x[:, i] = np.argmax(p[:, start:start+self.n_values[i]], axis=1)
            start += self.n_values[i]
        return self.to_onehot(x)


def normalize_prob(p, n_values: list):
    """Cap at 0, then normalize the probabilities for each feature"""
    assert len(p.shape) == 2, f'{len(p.shape)} != 2'
    p_ = np.maximum(0, p)
    start = 0
    for i, n in enumerate(n_values):
        s = np.sum(p_[:, start:start+n], axis=1)
        p_[:, start:start+n] /= np.stack([s] * n, axis=1)
        start += n
    return p_


def test_prob():
    np.random.seed(0)
    prob = Probabilities([2, 2, 3])

    print('\n original input')
    x = np.array([[0, 1, 2], [1, 1, 2]])
    print(f'{x.shape}')
    print(x)

    print('\nprobabilities')
    v = prob.to_onehot(x)
    print(f'{x.shape} -> {v.shape}')
    print(v)

    print('\nget back the original')
    x2 = prob.onehot_to_values(v)
    print(f'{v.shape} -> {x2.shape}')
    print(x2)

    print('\nprobabilities')
    p = np.random.random(size=(2, prob.length))
    p = prob.normalize(p)
    v = prob.prob_to_onehot(p)
    print(f'{p.shape} -> {v.shape}')
    print(np.round(p, 3))
    print(v)


if __name__ == '__main__':
    test_prob()
