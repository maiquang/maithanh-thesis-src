import numpy as np
from scipy.stats import multivariate_normal as mvn

from .statespace import StateSpace


class Trajectory:
    """Trajectory simulator class.

    Parameters
    ----------
    model : StateSpace object
        State-space representation model
    n_steps : int
        Number of simulation steps
    init_state : np.array, optional
        Initial state vector at t=0, default is a zero vector
    R : list of matrices/scalars, None, False, optional
        List of observation noise covariance matrices,
        by default False (use observation matrix from the initialized state-space model)
    random_seed : int, optional
        Random seed for PRNG initialization
    u : np.array, optional
        Control vector

    Attributes
    ----------
    model : StateSpace object
        State-space representation model
    states : np.array
        An array of simulated states
    observations : np.array
        An array of simulated observations

    Methods
    -------
    simulate(n_steps, init_state=None, random_seed=None, u=None)
        Simulate trajectory.
    add_obs_noise(R=False, random_seed=None)
        Add observation noise.
    """

    def __init__(
        self, model, n_steps, init_state=None, R=False, random_seed=None, u=None
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
        R : list of matrices/scalars, None, False, optional
            List of observation noise covariance matrices,
            by default False (use observation matrix from the initialized state-space model)
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

        if random_seed is not None:
            np.random.seed(random_seed)

        self.simulate(n_steps, init_state, u)
        if R is not None:
            self.add_obs_noise(R)

    def simulate(self, n_steps, init_state=None, u=None, random_seed=None):
        """Simulate a new trajectory, can be retrieves from the attribute "X"
        or "states".

        Parameters
        ----------
        n_steps : int
            Number of simulation steps
        init_state : np.array, optional
            Initial state vector at t=0, default is a zero vector
        u : np.array, optional
            Control vector
        random_seed : int, optional
            Random seed for PRNG initialization, by default None
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        X = np.zeros(shape=(n_steps, self.model.A.shape[0]))

        # Generate true states
        x = init_state
        for t in range(n_steps):
            x = self.model.A.dot(x) + mvn.rvs(cov=self.model.Q)
            if self.model.B is not None and u is not None:
                x += self.model.B.dot(u)
            X[t, :] = x

        self.X = X

    def add_obs_noise(self, R=False, random_seed=None):
        """Add observation noise to the generated trajectory. Noise is
        generated based on the function parameter R. Observations can be
        accessed from the attribute "observations" or "Y".

        Parameters
        ----------
        R : list of matrices/scalars, None, False, optional
            List of observation noise covariance matrices,
            by default False (use observation matrix from the initialized state-space model)
        random_seed : int, optional
            Random seed for PRNG initialization, by default None
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if R is False:
            # Use observation matrix from the initialized state-space model
            R = self.model.R

        if np.asarray(R).ndim == 1:
            # List of scalars (for 1D trajectory)
            R = np.asarray(R)[:, np.newaxis, np.newaxis]
        elif np.asarray(R).ndim == 2:
            # One matrix
            R = np.asarray(R)[np.newaxis, :]

        Y = np.zeros(
            shape=(self.X.shape[0], np.asarray(R).shape[0], self.model.H.shape[0])
        )

        # Add observation noise
        for i, r in enumerate(R):
            # y = Hx + obs.noise
            Y[:, i] = np.einsum("ij, kj -> ki", self.model.H, self.X) + mvn.rvs(
                cov=r, size=self.X.shape[0]
            ).reshape((self.X.shape[0], -1))

        self.Y = np.squeeze(Y)

    @property
    def states(self):
        return self.X

    @property
    def observations(self):
        return self.Y

    @states.setter
    def states(self, data):
        self.X = data

    @observations.setter
    def observations(self, data):
        self.Y = data
