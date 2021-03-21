import numpy as np
from scipy.stats import multivariate_normal as mvn

from .statespace import StateSpace


class Trajectory:
    """ Trajectory simulator class.

    Parameters
    ----------
    model : StateSpace object
        State-space representation model
    n_steps : int
        Number of simulation steps
    n_obs : int
        Number of distinct observations in each step
    init_state : np.array, optional
        Initial state vector at t=0, default is a zero vector
    random_seed : int, optional
        Random seed for PRNG initialization
    u : np.array, optional
        Control vector

    Attributes
    ----------
    model : StateSpace object
        State-space representation model
    X : np.array
        An array of simulated states
    Y : np.array
        An array of simulated observations

    Methods
    -------
    simulate(n_steps, init_state=None, random_seed=None, u=None)
        Simulate trajectory.
    """

    def __init__(
        self, model, n_steps, init_state=None, R=None, random_seed=None, u=None
    ):
        """Initialize trajectory simulator and simulate n_steps.

        Parameters
        ----------
        model : StateSpace object
            State-space representation model
        n_steps : int
            Number of simulation steps
        init_state : np.array, optional
            Initial state vector at t=0, default is a zero vector
        R :

        random_seed : int, optional
            Random seed for PRNG initialization
        u : np.array, optional
            Control vector
        """
        if not isinstance(model, StateSpace):
            raise TypeError(f"StateSpace object expected, got {type(model)}")

        self.model = model

        if init_state is None:
            init_state = np.zeros(self.model.A.shape[0], dtype=np.float64)

        self.X, self.Y = self.simulate(n_steps, init_state, R, u, random_seed)

    def simulate(self, n_steps, init_state=None, R=None, u=None, random_seed=None):
        """ Simulate trajectory.

        Parameters
        ----------
        model : StateSpace object
            State-space representation model
        n_steps : int
            Number of simulation steps
        n_obs : int
            Number of distinct observations in each step
        init_state : np.array, optional
            Initial state vector at t=0, default is a zero vector
        random_seed : int, optional
            Random seed for PRNG initialization
        u : np.array, optional
            Control vector

        Returns
        -------
        X : np.array
            An array of simulated states
        Y : np.array
            An array of simulated observations
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        if R is None:
            R = self.model.R

        if R.ndim == 1:
            # List of scalars
            R = np.atleast_3d(R)
        elif R.ndim == 2:
            # One matrix
            R = R[np.newaxis, :]

        X = np.zeros(shape=(n_steps, self.model.A.shape[0]))
        Y = np.zeros(shape=(n_steps, R.shape[0], self.model.H.shape[0]))

        # Generate true states
        x = init_state
        for t in range(n_steps):
            x = self.model.A.dot(x) + mvn.rvs(cov=self.model.Q)
            if self.model.B is not None and u is not None:
                x += self.model.B.dot(u)
            X[t, :] = x

        # Add observation noise
        for i, r in enumerate(R):
            # y = Hx + obs.noise
            Y[:, i] = np.einsum("ij, kj -> ki", self.model.H, X) + mvn.rvs(
                cov=r, size=n_steps
            )

        return X, np.squeeze(Y)

    @property
    def states(self):
        return self.X

    @property
    def observations(self):
        return self.Y
