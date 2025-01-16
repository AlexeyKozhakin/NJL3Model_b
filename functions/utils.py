import numpy as np


def midpoints(a, b, N):
    edges = np.linspace(a, b, N + 1)
    midp = (edges[:-1] + edges[1:]) / 2
    return midp