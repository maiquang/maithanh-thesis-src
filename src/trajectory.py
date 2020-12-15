import numpy as np
from scipy.stats import multivariate_normal as mvn

from .statespace import StateSpace


class Trajectory:
    """ Trajectory simulator class

    Parameters
    ----------
    model : StateSpace object
        State-space representation model
    n_steps : int
        Number of simulation steps
    init_state : ndarray, optional
        Initial state vector at t=0, default is a zero vector
    random_seed : int, optional
        Random seed for PRNG initialization
    u : ndarray, optional
        Control vector

    Attributes
    ----------
    model : StateSpace object
        State-space representation model
    X : ndarray
        An array of simulated states
    Y : ndarray
        An array of simulated observations
    """

    def __init__(self, model, n_steps, init_state=None, random_seed=None, u=None):
        """Initialize trajectory simulator and simulate n_steps.
        """
        if not isinstance(model, StateSpace):
            raise TypeError(f"StateSpace object expected, got {type(model)}")

        self.model = model

        if init_state is None:
            init_state = np.zeros(self.model.A.shape[0], dtype=np.float64)

        self.X, self.Y = self.simulate(n_steps, init_state, random_seed, u)

    def simulate(self, n_steps, init_state=None, random_seed=None, u=None):
        """ Simulate trajectory.

        Parameters
        ----------
        model : StateSpace object
            State-space representation model
        n_steps : int
            Number of simulation steps
        init_state : ndarray, optional
            Initial state vector at t=0, default is a zero vector
        random_seed : int, optional
            Random seed for PRNG initialization
        u : ndarray, optional
            Control vector

        Returns
        -------
        X : ndarray
            An array of simulated states
        Y : ndarray
            An array of simulated observations
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
    def observations(self):
        return self.Y


if __name__ == "__main__":
    pass
