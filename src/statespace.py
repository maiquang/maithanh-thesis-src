import numpy as np


class StateSpace:
    """
    Discrete time-invariant state-space representation.

        Parameters
    ----------
    A : np.array
        State transition matrix
    H : np.array
        Observation matrix
    Q : np.array
        Process noise covariance matrix
    R : np.array
        Observation noise covariance matrix
    B : np.array, optional
        Control transition matrix

    Attributes
    ----------
    A : np.array
        State transition matrix
    H : np.array
        Observation matrix
    Q : np.array
        Process noise covariance matrix
    R : np.array
        Observation noise covariance matrix
    B : np.array
        Control transition matrix
    """

    def __init__(self, A, H, Q, R, B=None):
        """
        Initialize custom state-space model representation.

        Parameters
        ----------
        A : np.array
            State transition matrix
        H : np.array
            Observation matrix
        Q : np.array
            Process noise covariance matrix
        R : np.array
            Observation noise covariance matrix
        B : np.array, optional
            Control transition matrix
        """
        self.A = A
        self.B = B
        self.Q = Q

        self.H = np.atleast_2d(H)
        self.R = np.atleast_2d(R)


class RWModel(StateSpace):
    """Random Walk Kinematic Model

    Parameters
    ----------
    q : float
        Process noise intensity
    r : float
        Observation noise variance
    ndim : int
        Number of observed position dimensions
    dt : int
        Sampling period
    """

    def __init__(self, q, r, ndim=2, dt=1):
        """Initialize random walk model.

        Parameters
        ----------
        q : float
            Process noise intensity
        r : float
            Observation noise variance
        ndim : int
            Number of observed position dimensions
        dt : int
            Sampling period
        """
        A = np.eye(ndim)
        Q = q * (dt * np.eye(ndim))

        H = np.eye(ndim)
        R = (r ** 2) * np.eye(ndim)
        super().__init__(A, H, Q, R)


class CVModel(StateSpace):
    """Constant Velocity Kinematic Model

    Parameters
    ----------
    q : float
        Process noise intensity
    r : float
        Observation noise variance
    ndim : int
        Number of observed position dimensions
    dt : int
        Sampling period
    """

    def __init__(self, q, r, ndim=2, dt=1):
        """Initialize constant velocity model.

        Parameters
        ----------
        q : float
            Process noise intensity
        r : float
            Observation noise variance
        ndim : int
            Number of observed position dimensions
        dt : int
            Sampling period
        """
        A = np.eye(2 * ndim) + dt * np.eye(2 * ndim, k=ndim)

        Q = np.zeros((2 * ndim, 2 * ndim))
        for i in range(ndim):
            Q[i, i] = (dt ** 3) / 3
            Q[i, i + ndim] = (dt ** 2) / 2

            Q[i + ndim, i] = (dt ** 2) / 2
            Q[i + ndim, i + ndim] = dt
        Q = q * Q

        H = np.eye(ndim, 2 * ndim)
        R = (r ** 2) * np.eye(ndim)

        super().__init__(A, H, Q, R)


class CAModel(StateSpace):
    """Constant Acceleration Kinematic Model

    Parameters
    ----------
    q : float
        Process noise intensity
    r : float
        Observation noise variance
    ndim : int
        Number of observed position dimensions
    dt : int
        Sampling period
    """

    def __init__(self, q, r, ndim=2, dt=1):
        """Initialize constant acceleration model.

        Parameters
        ----------
        q : float
            Process noise intensity
        r : float
            Observation noise variance
        ndim : int
            Number of observed position dimensions
        dt : int
            Sampling period
        """
        A = (
            np.eye(3 * ndim)
            + dt * np.eye(3 * ndim, k=ndim)
            + (dt ** 2) / 2 * np.eye(3 * ndim, k=2 * ndim)
        )

        Q = np.zeros((3 * ndim, 3 * ndim))
        for i in range(ndim):
            Q[i, i] = (dt ** 5) / 20
            Q[i, i + ndim] = (dt ** 4) / 8
            Q[i, i + 2 * ndim] = (dt ** 3) / 6

            Q[i + ndim, i] = (dt ** 4) / 8
            Q[i + ndim, i + ndim] = (dt ** 3) / 3
            Q[i + ndim, i + 2 * ndim] = (dt ** 2) / 2

            Q[i + 2 * ndim, i] = (dt ** 3) / 6
            Q[i + 2 * ndim, i + ndim] = (dt ** 2) / 2
            Q[i + 2 * ndim, i + 2 * ndim] = dt
        Q = q * Q

        H = np.eye(ndim, 3 * ndim)
        R = (r ** 2) * np.eye(ndim)
        super().__init__(A, H, Q, R)
