import numpy as np
from scipy.stats import multivariate_normal as mvn

from .statespace import StateSpace


class Trajectory:
    """ Trajectory simulator class

    Parameters
    ----------
    model : StateSpace object
        State-space representation model
    init_state : ndarray
        Initial state at t=0
    n_steps : int
        Number of simulation steps
    random_seed : int, optional
        Random seed for PRNG initialization
    u : ndarray, optional
        Input vector

    Attributes
    ----------
    model : StateSpace object
        State-space representation model
    X : ndarray
        An array of simulated states
    Y : ndarray
        An array of simulated measurements
    """

    def __init__(self, model, init_state, n_steps, random_seed=None, u=None):
        """Initialize trajectory simulator and simulate n_steps.
        """
        if not isinstance(model, StateSpace):
            raise TypeError(f"StateSpace object expected, got {type(model)}")

        self.model = model
        self.X, self.Y = self.simulate(init_state, n_steps, random_seed, u)

    def simulate(self, init_state, n_steps, random_seed=None, u=None):
        """ Simulate trajectory.

        Parameters
        ----------
        model : StateSpace object
            State-space representation model
        init_state : ndarray
            Initial state at t=0
        n_steps : int
            Number of simulation steps
        random_seed : int, optional
            Random seed for PRNG initialization
        u : ndarray, optional
            Input vector

        Returns
        -------
        X : ndarray
            An array of simulated states
        Y : ndarray
            An array of simulated measurements

        """

        if random_seed is not None:
            np.random.seed(random_seed)

        X = np.zeros(shape=(n_steps, self.model.A.shape[0]))
        Y = np.zeros(shape=(n_steps, self.model.H.shape[0]))

        x = init_state
        for t in range(n_steps):
            x = self.model.A.dot(x) + mvn.rvs(cov=self.model.Q)
            if self.model.B is not None and u is not None:
                x += self.model.B.dot(u)

            y = self.model.H.dot(x) + mvn.rvs(cov=self.model.R)

            X[t, :] = x
            Y[t, :] = y

        return X, Y

    @property
    def states(self):
        return self.X

    @property
    def measurements(self):
        return self.Y


if __name__ == "__main__":
    pass
