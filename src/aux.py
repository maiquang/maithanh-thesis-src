import numpy as np
from matplotlib import pyplot as plt


def rmse(y, x, n):
    """ Compute time series RMSE.

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
    if y.ndim == 1:
        y = np.expand_dims(y, 1)

    if x.ndim == 1:
        x = np.expand_dims(x, 1)

    t = np.arange(1, n + 1)
    return np.sqrt(((y[:n] - x[:n]) ** 2).cumsum(axis=0) / t.reshape(n, 1))


def plot_traj(traj, obs=None, kf=None):
    """ Plot 2D trajectory.

    Parameters
    ----------
    traj : Trajectory object
        Simulated trajectory
    obs : int, optional
        Which set of observations to plot
    kf : KalmanFilter object, optional
        KalmanFilter estimator, plot estimates
    """
    plt.figure(figsize=(10, 10))
    # plt.axis("equal")
    plt.plot(
        traj.X[:, 0],
        traj.X[:, 1],
        label="True value",
        alpha=0.4 if kf is not None else 1.0,
    )

    if obs is not None:
        if traj.Y.ndim == 3:
            plt.plot(
                traj.Y[:, obs, 0],
                traj.Y[:, obs, 1],
                "+",
                label="Observations",
                alpha=0.5,
            )
        else:
            # Only one set of observations exists
            plt.plot(
                traj.Y[:, 0], traj.Y[:, 1], "+", label="Observations", alpha=0.5,
            )
    if kf is not None:
        hist_est = kf.history
        plt.plot(
            hist_est[:, 0],
            hist_est[:, 1],
            linestyle="dotted",
            label="State estimate",
            color="r",
        )

    plt.xlabel(f"$x_1$", fontsize="large")
    plt.ylabel(f"$x_2$", fontsize="large")

    plt.legend()
    plt.title("Trajectory")
    plt.show()


def plot_rmse(traj, *kfs, nvars="all", var_labels=None, kf_labels=None):
    """ Plot the evolution of RMSE.

    Parameters
    ----------
    traj: Trajectory object
        Simulated trajectory
    kfs: KalmanFilter objects
        KalmanFilter estimator
    nvars : int, default "all"
        How many variables to plot
    var_labels : list, optional
        Variable labels
    kf_labels : list, optional
        List of size len(kfs), KalmanFilter labels
    """
    nvars = nvars if nvars != "all" else traj.X.shape[1]
    ndat = len(traj.X)

    plt.figure(figsize=(12, nvars * 5))
    for i in range(nvars):
        plt.subplot(nvars, 1, i + 1)

        for j in range(len(kfs)):
            hist_est = kfs[j].history
            if hist_est.shape[1] - 1 < i:
                continue
            rmse_i = rmse(traj.X[:, i], hist_est[:, i], ndat)
            try:
                plt.plot(rmse_i, label=kf_labels[j])
            except (IndexError, TypeError):
                plt.plot(rmse_i, label=f"kf-{j}")

        try:
            plt.ylabel(var_labels[i], fontsize="large")
        except (TypeError, IndexError):
            plt.ylabel(f"$var_{i+1}$", fontsize="large")

        # plt.ylim(bottom=0.0)
        plt.xlabel("$t$", fontsize="large")
        plt.grid(linestyle="dotted")
        plt.legend()

    plt.suptitle("Root mean squared error")
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.show()


def plot_estimates(kf, traj, nvars="all", plt_std=True, labels=None):
    """ Plot the evolution of estimates from kf.

    Parameters
    ----------
    kf : KalmanFilter object
        KalmanFilter estimator
    traj : Trajectory object
        Simulated trajectory
    nvars : int, default "all"
        How many variables to plot
    plt_std : bool, default True
        Plot +/- 3 std bands
    labels : list
        Variable labels
    """
    hist_est = kf.history
    hist_cov = kf.history_cov
    nvars = nvars if nvars != "all" else hist_est.shape[1]

    plt.figure(figsize=(12, nvars * 7))
    for i in range(nvars):
        plt.subplot(nvars, 1, i + 1)
        plt.plot(traj.X[:, i], alpha=0.6, color="b")
        plt.plot(hist_est[:, i], ":", color="r")
        if plt_std:
            std = np.sqrt(hist_cov[:, i, i])
            upper = hist_est[:, i] + 3 * std
            lower = hist_est[:, i] - 3 * std
            plt.plot(upper, "--", alpha=0.2, color="b")
            plt.plot(lower, "--", alpha=0.2, color="r")

            # plt.fill_between(
            #     x=np.arange(0, len(hist_est), 1), y1=upper, y2=lower, alpha=0.05,
            # )

        ymin = np.min([hist_est[:, i], traj.X[:, i]]) - 10
        ymax = np.max([hist_est[:, i], traj.X[:, i]]) + 10
        plt.vlines(
            kf._reset_log, ymin, ymax, colors="g", linestyles="dashed", alpha=0.3,
        )

        plt.ylim((ymin, ymax))

        try:
            plt.ylabel(labels[i], fontsize="large")
        except (TypeError, IndexError):
            plt.ylabel(f"$var_{i+1}$", fontsize="large")
        plt.xlabel("$t$", fontsize="large")

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.figlegend(["True value", "State estimate", "Filter reset"])
    plt.suptitle("Evolution of estimates")
    plt.show()
