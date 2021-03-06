import numpy as np
from matplotlib import pyplot as plt


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


def plot_estimates(est, traj, n=None, labels=None):
    hist = est.history
    n = n if n is not None else hist.shape[1]

    plt.figure(figsize=(12, n * 4))
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(hist[:, i])
        plt.plot(traj.X[:, i])
        plt.xlabel("$t$", fontsize="large")
        plt.vlines(
            est._reset_log,
            np.min([hist[:, i], traj.X[:, i]]),
            np.max([hist[:, i], traj.X[:, i]]),
            colors="g",
            linestyles="dashed",
            alpha=0.7,
        )

        try:
            plt.ylabel(labels[i])
        except:
            pass

    plt.figlegend(["Estimate", "Real value", "Filter reset"])
    plt.show()
