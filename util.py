import numpy as np


def generate_dataset(N=100, T=100):
    xs, ys = [], []

    for n in range(N):
        x = np.arange(0, T)
        y = np.sin(2.0 * np.pi * x / T) + 0.05 * \
            np.random.uniform(-1.0, 1.0, T)

        xs.append(x)
        ys.append(y)
    xs = np.array(xs).reshape(N, -1, T)
    ys = np.array(ys).reshape(N, -1, T)

    return xs, ys
