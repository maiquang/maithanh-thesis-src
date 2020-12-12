import numpy as np
from scipy.stats import multivariate_normal as mvn

from statespace import CAModel


class Trajectory:
    """ Trajectory simulator class

    Attributes
    ----------
    model : StateSpace object

    X : ndarray

    Y : ndarray

    """

    def __init__(self, model, init_state, n_steps, random_state=None, u=None):
        """ Initialize trajectory simulator and simulate n_steps.

        Parameters
        ----------
        model : StateSpace object

        init_state: vector

        n_steps : int

        random_state : int

        u : vector
            input vector

        """
        self.model = model
        self.X, self.Y = self.simulate(init_state, n_steps, random_state, u)

    def simulate(self, init_state, n_steps, random_state=None, u=None):
        """ Simulate trajectory.

        """

        if random_state is not None:
            np.random.seed(random_state)

        X = np.zeros(shape=(n_steps, self.model.A.shape[0]))
        Y = np.zeros(shape=(n_steps, self.model.H.shape[0]))

        x = init_state
        for t in range(n_steps):
            x = self.model.A.dot(x) + mvn.rvs(cov=self.model.Q)
            if self.model.B is not None and u is not None:
                x += self.model.B @ u

            y = self.model.H.dot(x) + mvn.rvs(cov=self.model.R)

            X[t, :] = x
            Y[t, :] = y

        return X, Y


if __name__ == "__main__":
    cam = CAModel(1, 1)
    traj = Trajectory(cam, np.ones(6), 100)

    print(traj.X.shape)
    print(traj.Y.shape)
