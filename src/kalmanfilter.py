import numpy as np
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

    Attributes
    ----------
    model : StateSpace object
        State-space representation model
    nbh : list of references to KalmanFilter objects
        Agents in the neighborhood of the this agent, includes this agent
        at the first position
    x : np.array
        Current state estimate
    P : np.array
        Current state covariance matrix
    history : np.array
        History of state estimates

    Methods
    -------
    predict(u=None, log=False)
        Predict the next state using the state-space equation.
    update(y, R=None, log=False)
        Correct/update the current state estimate.
    get_estimate(indices=None)
        Get current state estimate.
    add_nbhs(*kfs)
        Add agents to this agent's neighborhood.
    get_nbh_estimates(indices=None)
        Get estimates from agents in neighborhood.
    cov_intersect(weights=None, normalize=True, log=True)
        Calculate combined estimate from neighborhood agents' estimates.
    """

    def __init__(self, model, x0=None, P0=None):
        """ Initialize Kalman Filter.
        """
        if not isinstance(model, StateSpace):
            raise TypeError(f"StateSpace object expected, got {type(model)}")
        self.model = model
        self.nbh = [self]

        self._ndim = self.model.A.shape[0]

        # Save priors in case a reset is needed
        self.x0 = x0 if x0 else np.zeros(self._ndim)
        self.P0 = P0 if P0 else np.eye(self._ndim) * 1000

        # Priors
        self.x = x0 if x0 else np.zeros(self._ndim)
        self.P = P0 if P0 else np.eye(self._ndim) * 1000

        self._I = np.eye(self._ndim)  # for update() method
        self._history = []  # logging

    def predict(self, u=None, lambda_expf=None, log=False):
        """ Predict the next state using the state-space equation.

        Parameters
        ----------
        u : np.array, optional
            Control vector
        lambda_expf : float, optional
            Exponential forgetting parameter, should be from the interval (0, 1)
        log : bool, default True
            Log the resulting state estimate
        """
        xminus = self.model.A.dot(self.x)
        if u is not None:
            xminus += self.model.B.dot(u)

        if lambda_expf is not None:
            if 0 < lambda_expf < 1:
                self.P *= 1 / lambda_expf
            else:
                raise ValueError(
                    f"Exponential forgetting parameter lamba_expf is not from the interval (0, 1)."
                )
        Pminus = self.model.A.dot(self.P).dot(self.model.A.T) + self.model.Q

        self.x = xminus
        self.P = Pminus

        if log:
            self._log()

    def update(self, y, R=None, log=False):
        """ Correct/update the current state estimate.

        Parameters
        ----------
        y : np.array
            Observation for this update
        R : np.array, optional
            Override observation noise matrix for this step
        log : bool, default True
            Log the resulting state estimate
        """
        if R is None:
            R = self.model.R
        H = self.model.H

        PHT = self.P.dot(H.T)
        # K = PH'(HPH' + R)^-1
        K = np.linalg.inv(H.dot(PHT) + R)
        K = PHT.dot(K)

        # x+ = x-  +K(y - Hx-)
        innov = y - H.dot(self.x)
        xplus = self.x + K.dot(innov)

        # P+ = (I-KH)P-(I-KH)' + KRK'
        # Should support non-optimal K
        I_KH = self._I - K.dot(H)
        KRK = K.dot(R).dot(K.T)
        Pplus = I_KH.dot(self.P).dot(I_KH.T) + KRK

        self.x = xplus
        self.P = Pplus

        if log:
            self._log()

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

    def add_nbhs(self, *kfs):
        """ Add agents to this agent's neighborhood.

        Parameters
        ----------
        *kfs : KalmanFilter objects
            One or more agents to add to this agent's neighborhood
        """
        for kf in kfs:
            self.nbh.append(kf)

    def get_nbh_estimates(self, indices=None):
        """ Get estimates from agents in neighborhood. This function should be
        called after predict() and update() but before cov_intersect().

        Parameters
        ----------
        indices : list-of-lists/arrays, optional
            A list, the size of neighborhood, of indices of variables over
            which to get marginal distributions for each agent in neighborhood

            Selects ndim first variables by default, ndim is the dimension of
            this agent's estimate
        """
        if not indices:
            # Default case
            indices = [None for i in range(len(self.nbh))]

        self._nbh_ests = []
        for agent, i in zip(self.nbh, indices):
            # Only consider estimates from more complex models
            if agent._ndim >= self._ndim:
                # Default case if i is None
                self._nbh_ests.append(
                    agent.get_estimate(
                        indices=np.arange(self._ndim, dtype=np.int) if not i else i
                    )
                )

    def cov_intersect(self, weights=None, normalize=True, log=True):
        """ Calculate combined estimate from neighborhood agents' estimates
        using the covariance intersection algorithm. Updated neighborhood
        estimates should be obtained by calling get_nbh_estimates()
        before each cov_intersect() call.

        Parameters
        ----------
        weights : list-like
            Weights for each agent in the neighborhood
        normalize : bool, optional, default True
            Normalize weights
        log : bool, default True
            Log the resulting state estimate
        """
        if len(self._nbh_ests) < 2:
            return

        if weights is None:
            weights = np.ones(shape=len(self._nbh_ests))
            weights /= np.sum(weights)

        if normalize is True:
            weights = np.asarray(weights) / np.sum(weights)

        P_newinv = np.zeros_like(self.P)
        xnew = np.zeros_like(self.x)

        for w, (x, P) in zip(weights, self._nbh_ests):
            Pinv = np.linalg.inv(P)
            P_newinv += w * Pinv
            xnew += w * Pinv.dot(x.T)

        self.P = np.linalg.inv(P_newinv)
        self.x = self.P.dot(xnew)

        if log:
            self._log()

    def reset_filter(self, dist_thresh=None):
        """
        """
        # Compute mean pos. estimate
        # Reset
        # Init with default state or custom default state
        pass

    def _log(self):
        self._history.append(self.x.copy())

    @property
    def history(self):
        return np.asarray(self._history)
