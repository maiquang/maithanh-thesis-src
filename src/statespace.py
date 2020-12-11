import numpy as np


class StateSpace:
    def __init__(self, A, H, Q, R, B=None):
        self.A = A
        self.B = B
        self.Q = Q

        self.H = H
        self.R = R


class RWModel(StateSpace):
    def __init__(self, q, r, dt=1, ndim=2):
        # Random Walk Model
        # fmt:off
        A = np.eye(ndim)
        Q = q * (dt * np.eye(ndim))

        H = np.eye(ndim)
        R = r**2 + np.eye(ndim)
        # fmt:on
        super().__init__(A, H, Q, R)


class CVModel(StateSpace):
    def __init__(self, q, r, dt=1, ndim=2):
        # Constant Velocity Model
        # fmt:off
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])

        Q = q * np.array([[dt**3/3, 0      , dt**2/2, 0      ],
                          [0,       dt**3/3, 0,       dt**2/2],
                          [dt**2/2, 0,       dt,      0      ],
                          [0,       dt**2/2, 0,       dt     ]])

        H = np.array([[1., 0, 0, 0],
                      [0., 1, 0, 0]])
        R = r**2 * np.eye(2)
        # fmt:on
        super().__init__(A, H, Q, R)


class CAModel(StateSpace):
    def __init__(self, q, r, dt=1, ndim=2):
        # Constant Acceleration Model
        # fmt:off
        A = np.array([
            [1, 0, dt,  0, dt**2/2,       0],
            [0, 1, 0,  dt,       0, dt**2/2],
            [0, 0, 1,   0,      dt,       0],
            [0, 0, 0,   1,       0,      dt],
            [0, 0, 0,   0,       1,       0],
            [0, 0, 0,   0,       0,       1]
        ])

        Q = q * np.array([
            [dt**5/20,        0, dt**4/8,       0, dt**3/6,       0],
            [       0, dt**5/20,       0, dt**4/8,       0, dt**3/6],
            [ dt**4/8,        0, dt**3/3,       0, dt**2/2,       0],
            [       0,  dt**4/8,       0, dt**3/3,       0, dt**2/2],
            [ dt**3/6,        0, dt**2/2,       0,      dt,       0],
            [       0,  dt**3/6,       0, dt**2/2,       0,      dt]

        ])

        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]])
        R = r**2 * np.eye(2)
        # fmt: on
        super().__init__(A, H, Q, R)
