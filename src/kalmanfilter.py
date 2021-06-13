import numpy as np
import pandas as pd

from itertools import repeat, chain

from pandas.core.arrays import boolean
from .statespace import StateSpace


class KalmanFilter:
    """ Kalman Filter class

    Parameters
    ----------
    model : StateSpace object
        State-space representation model
    x0 : np.array, optional
        Prior state estimate
    P : np.array, optional
        Prior state covariance matrix
    lambda_expf : float, optional
        Exponential forgetting parameter, should be from the interval (0, 1)

    Attributes
    ----------
    model : StateSpace object
        State-space representation model
    x : np.array
        Current state estimate
    P : np.array
        Current state covariance matrix
    history : np.array
        History of state estimates
    history_cpv : np.array
        History of covariance matrices of estimates

    Methods
    -------
    predict(u=None, log=False)
        Predict the next state using the state-space equation.
    update(y, R=None, log=False)
        Correct/update the current state estimate.
    set_estimate(xnew, P_new)
        Update/override the current estimate.
    get_estimate(indices=None)
        Get current state estimate.
    get_observation()
        Return the latest observation.
    """

    def __init__(self, model, x0=None, P0=None, lambda_expf=1.0):
        """ Initialize Kalman Filter.

        Parameters
        ----------
        model : StateSpace object
            State-space representation model
        x0 : np.array, optional
            Prior state estimate
        P : np.array, optional
            Prior state covariance matrix
        lambda_expf : float, optional
            Exponential forgetting parameter, should be from the interval (0, 1)
        """
        if not isinstance(model, StateSpace):
            raise TypeError(f"StateSpace object expected, got {type(model)}")

        if not (0 < lambda_expf <= 1.0):
            raise ValueError(
                f"Exponential forgetting parameter lamba_expf={lambda_expf} is not from the interval (0, 1]."
            )

        self.model = model
        self.nbh = [self]
        self.lambda_expf = lambda_expf

        self._ndim = self.model.A.shape[0]

        # Initialize current estimates with priors
        self.x = x0 if x0 is not None else np.zeros(self._ndim)
        self.P = P0 if P0 is not None else np.eye(self._ndim) * 1000

        self.y = None  # latest observation

        # Save priors in case a reset is needed
        self._x0 = x0 if x0 is not None else np.zeros(self._ndim)
        self._P0 = P0 if P0 is not None else np.eye(self._ndim) * 1000

        self._I = np.eye(self._ndim)  # for update() method

        self._history_est = []  # logging
        self._history_cov = []
        self._reset_log = []

    def predict(self, u=None, log=False):
        """ Predict the next state using the state-space equation.

        Parameters
        ----------
        u : np.array, optional
            Control vector
        log : bool, default True
            Log the resulting state estimate
        """
        self.P *= 1 / self.lambda_expf

        xminus = self.model.A.dot(self.x)
        if u is not None:
            xminus += self.model.B.dot(u)
        Pminus = self.model.A.dot(self.P).dot(self.model.A.T) + self.model.Q

        self.x = xminus
        self.P = Pminus

        if log:
            self._log()

    def update(self, y, R=None, w=1.0, log=False):
        """ Correct/update the current state estimate.

        Parameters
        ----------
        y : np.array
            Observation for this update
        R : np.array, optional
            Override observation noise matrix for this step
        w : float
            Measurement weight from the interval [0.0, 1.0]
        log : bool, default True
            Log the resulting state estimate
        """
        if R is None:
            R = self.model.R

        if y is None:
            # No observation is available
            self.y = None
            return

        H = self.model.H
        PHT = self.P.dot(H.T)
        # K = PH'(HPH' + R)^-1
        K = np.linalg.inv(w * H.dot(PHT) + R)
        K = PHT.dot(K)

        # x+ = x- + K(y - Hx-)
        innov = y - H.dot(self.x)
        xplus = self.x + K.dot(innov)

        # P+ = (I-KH)P- ~ used by most implementations
        # P+ = (I-KH)P-(I-KH)' + KRK' ~ numericaaly more stable
        I_KH = self._I - K.dot(H)
        KRK = K.dot(R).dot(K.T)
        Pplus = I_KH.dot(self.P).dot(I_KH.T) + KRK

        self.x = xplus
        self.P = Pplus
        self.y = y

        if log:
            self._log()

    def set_estimate(self, xnew, P_new):
        """ Update/override the current estimate.

        Parameters
        ----------
        xnew : np.array
            New state estimate x
        P_new : np.array
            New covariance matrix P
        """
        self.x = xnew
        self.P = P_new

    def get_estimate(self, indices=None):
        """ Get current state estimate.

        Parameters
        ----------
        indices : list-like, optional
            Indices of variables over which to get marginal distributions

        Returns
        -------
        (x, P) : (np.array, np.array)
            Current state estimate x and covariance matrix P
        """
        if indices is None:
            indices = np.arange(self._ndim, dtype=np.int)

        return (self.x[indices], self.P[np.ix_(indices, indices)])

    def get_observation(self):
        """ Return the latest observation.

        Returns
        -------
        y : np.array
            Latest local observation from update()
        R : np.array
            Observation noise matrix
        """
        return self.y, self.model.R

    def reset_filter(self, init_state=None, init_cov=None):
        """ Reset the filter and initialize with init_state and init_cov.
        Log the reset to _reset_log.

        Parameters
        ----------
        init_state : np.array
            New state estimate x
        init_cov : np.array
            New covariance matrix P
        """
        self.set_estimate(
            init_state if init_state is not None else self._x0,
            init_cov if init_cov is not None else self._P0,
        )

        self._reset_log.append(len(self._history_est))

    def _log(self):
        self._history_est.append(self.x.copy())
        self._history_cov.append(self.P.copy())

    def to_dataframe(self, traj):
        """Generate a pandas DataFrame of past estimates.

        Parameters
        ----------
        traj : Trajectory object
            ...

        Returns
        -------
        DataFrame
            A pandas DataFrame containing past estimates and real states
        """
        # Get covariance and reset data
        std = np.sqrt(np.diagonal(self.history_cov, axis1=1, axis2=2))
        resets = np.zeros(len(self.history))
        resets[self._reset_log] = 1

        # Merge into one ndarray
        data = np.hstack((self.history, std, traj.states, resets[:, np.newaxis]))

        # Create column labels
        cols = []
        for i in range(1, self.history.shape[1] + 1):
            cols.append(f"x{i}_est")

        for i in range(1, self.history.shape[1] + 1):
            cols.append(f"x{i}_std")

        for i in range(1, traj.states.shape[1] + 1):
            cols.append(f"x{i}_real")
        cols.append("reset")

        return pd.DataFrame(data=data, columns=cols)

    @property
    def history(self):
        return np.asarray(self._history_est)

    @property
    def history_cov(self):
        return np.asarray(self._history_cov)
