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
    lambda_expf : float, optional
        Exponential forgetting parameter, should be from the interval (0, 1)

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

    def __init__(self, model, x0=None, P0=None, lambda_expf=None):
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

        if lambda_expf is not None and (not (0 < lambda_expf < 1)):
            raise ValueError(
                f"Exponential forgetting parameter lamba_expf is not from the interval (0, 1)."
            )

        self.model = model
        self.nbh = [self]
        self.lambda_expf = lambda_expf

        self._ndim = self.model.A.shape[0]

        # Initialize current estimates with priors
        self.x = x0 if x0 else np.zeros(self._ndim)
        self.P = P0 if P0 else np.eye(self._ndim) * 1000

        self.y = None  # latest observation

        # Save priors in case a reset is needed
        self._x0 = x0 if x0 else np.zeros(self._ndim)
        self._P0 = P0 if P0 else np.eye(self._ndim) * 1000

        self._I = np.eye(self._ndim)  # for update() method
        self._history = []  # logging
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
        xminus = self.model.A.dot(self.x)
        if u is not None:
            xminus += self.model.B.dot(u)

        if self.lambda_expf is not None:
            self.P *= 1 / self.lambda_expf

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
        """ Return latest observation.
        """
        return self.y

    def add_nbhs(self, *kfs):
        """ Add agents to this agent's neighborhood.

        Parameters
        ----------
        *kfs : KalmanFilter objects
            One or more agents to add to this agent's neighborhood
        """
        for kf in kfs:
            self.nbh.append(kf)

    def get_nbh_estimates(self, reset_thresh=None, indices=None):
        """ Get estimates from agents in neighborhood. This function should be
        called after predict() and update() but before cov_intersect().

        Parameters
        ----------
        reset_thresh : float, optional
            Maximum accepted distance from the centroid before the filter reset
            If None then filter is never reset
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
            # Only consider estimates from models with same/higher complexity
            if agent._ndim >= self._ndim:
                # i is usually None -> select first _ndim variables
                self._nbh_ests.append(
                    agent.get_estimate(
                        indices=np.arange(self._ndim, dtype=np.int) if not i else i
                    )
                )

        if reset_thresh is not None:
            self._reset_filter(reset_thresh)

    def cov_intersect(self, weights=None, normalize=True, log=False):
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

    def _reset_filter(self, reset_thresh, init_state=None, init_cov=None):
        """ Reset filter if Euclidean distance from the centroid
        is larger than reset_thresh.

        Centroid is calculated as the arithmetic mean of
        neighborhood estimates (self excluded).

        Parameters
        ----------
        reset_thresh : float
            Maximum accepted distance from the centroid
        init_state : np.array, optional
            State estimate after reset, by default it is the centroid
        init_cov : np.array, optional
            State covariance matrix after reset
        """
        if not isinstance(reset_thresh, (float, int)):
            raise TypeError(f"Number expected, got {type(reset_thresh)}")

        if len(self._nbh_ests) < 3:
            return

        ctr = np.asarray([x for (x, P) in self._nbh_ests[1:]]).mean(axis=0)
        dist_from_ctr = np.linalg.norm(ctr - self.x)
        if dist_from_ctr >= reset_thresh:
            # self.x = init_state if init_state else self._x0
            self.x = init_state if init_state else ctr
            self.P = init_cov if init_cov else self._P0

            self._nbh_ests[0] = self.get_estimate()

    def reset_filter(self, init_state=None, init_cov=None):
        self.set_estimate(
            init_state if init_state is not None else self._x0,
            init_cov if init_cov is not None else self._P0,
        )

        self._reset_log.append(len(self._history))

    def _log(self):
        self._history.append(self.x.copy())

    @property
    def history(self):
        return np.asarray(self._history)
