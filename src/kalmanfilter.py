import numpy as np
from .statespace import StateSpace


class KalmanFilter:
    def __init__(self, model, x0=None, P=None):
        if not isinstance(model, StateSpace):
            raise TypeError(f"StateSpace object expected, got {type(model)}")
        self.model = model

        # Priors
        self.P = P if P else np.eye(self.model.A.shape[0]) * 1000
        self.x = x0 if x0 else np.zeros(self.model.A.shape[0])

        self._I = np.eye(self.model.A.shape[0])
        self._history = []

    def predict(self, u=None, log=False):
        xminus = self.model.A.dot(self.x)
        if u is not None:
            xminus += self.model.B.dot(u)
        Pminus = self.model.A.dot(self.P).dot(self.model.A.T) + self.model.Q
        self.x = xminus
        self.P = Pminus

        if log:
            self._log()

    def update(self, y, R=None, log=True):
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