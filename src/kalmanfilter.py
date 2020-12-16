import numpy as np
from .statespace import StateSpace


class KalmanFilter:
    """ Kalman Filter class.

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
    x : np.array
        Current state estimate
    P : np.array
        Current state covariance matrix
    history : np.array
        History of state estimates
    """

    def __init__(self, model, x0=None, P=None):
        """ Initialize Kalman Filter.
        """
        if not isinstance(model, StateSpace):
            raise TypeError(f"StateSpace object expected, got {type(model)}")
        self.model = model

        # Priors
        self.x = x0 if x0 else np.zeros(self.model.A.shape[0])
        self.P = P if P else np.eye(self.model.A.shape[0]) * 1000

        self._I = np.eye(self.model.A.shape[0])   # for update() method
        self._history = []   # logging

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
        Pminus = self.model.A.dot(self.P).dot(self.model.A.T) + self.model.Q
        self.x = xminus
        self.P = Pminus

        if log:
            self._log()

    def update(self, y, R=None, log=True):
        """ Correct/update the current state estimate

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

        # x+ = x- + K(y - Hx-)
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

    def cov_intersect(self):
        pass

    def _log(self):
        self._history.append(self.x.copy())

    @property
    def history(self):
        return np.asarray(self._history)
