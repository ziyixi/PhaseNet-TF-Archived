import numpy as np


def normalize_range(data, xmin, xmax):
    return np.interp(data, (data.min(), data.max()), (xmin, xmax))
