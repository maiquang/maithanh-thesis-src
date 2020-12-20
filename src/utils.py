import numpy as np


def rmse(y, x, n):
    """ Calculate time series RMSE.

    Parameters
    ----------
    y : np.array
        Array of n true states
    x : np.array
        Array of n estimates
    n : int
        Number of time steps

    Returns
    -------
    np.array : RMSE array at every time step
    """
    t = np.arange(1, n + 1)
    return np.sqrt(((y[:n, 0:2] - x[:n, 0:2]) ** 2).cumsum(axis=1) / t.reshape(n, 1))
