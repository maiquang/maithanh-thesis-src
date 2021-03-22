import copy
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from .kalmanfilter import KalmanFilter
from .statespace import RWModel, CVModel, CAModel


class KFNet:
    """ Implements communication between KalmanFilter nodes.
    Uses networkx.expected_degree_graph() to generate the network.

    Parameters
    ----------
    nodes : int
        Number of nodes to be generated, ignored if G is provided
    avg_deg : int
        Average degree, however, the generator might not produce this exact
        average degree, ignored if G is provided
    init : list or dict
        List of KalmanFilter objects (or dict of {int : KalmanFilter} pairs)
        to be assigned to nodes
    txt_labels : list or dict
        List of labels (or dict of {int : string} pairs) to be assigned to nodes
    w_adapt : 2D np.array
        Adaptation weights matrix
    w_combine : 2D np.array
        Combination weights matrix
    random_seed : int, optional
        Random seed for the graph generator
    G : networkx.Graph, np.array
        A custom graph or an adjacency matrix

    Attributes
    ----------
    adj_mat :

    nnodes :

    kfs :

    w_adapt :

    w_combine :

    Methods
    -------
    assign(init=None, txt_labels=None)
        Assign KalmanFilter objects to nodes.
    generate_txt_labels()
        Generate text labels for nodes based on their motion model.
    draw_network(self, node_size, figsize)
        Draw network.
    predict(u)
        Run KalmanFilter predict step on all nodes.
    update(y)
        Run KalmanFilter update step on all nodes.
    adapt()
        Adaptation step - incorporates observations from neighbor nodes
        using update().
    combine()
        Combination step - combines estimates from neighbors using
        covariance intersection algorithm.
    time_step(predict, update, adapt, combine, reset_strategy, reset_thresh=5.0)
        Runs one iteration of diffusion Kalman filtering
        using adapt-then-combine strategy.
    """

    def __init__(
        self,
        nodes=15,
        avg_deg=10,
        init=None,
        txt_labels=None,
        w_adapt=None,
        w_combine=None,
        random_seed=None,
        G=None,
    ):
        """ Generate and initialize network.

        Parameters
        ----------
        nodes : int
            Number of nodes to be generated, ignored if G is provided
        avg_deg : int
            Average degree, however, the generator might not produce this exact
            average degree, ignored if G is provided
        init : list or dict
            List of KalmanFilter objects (or dict of {int : KalmanFilter} pairs)
            to be assigned to nodes
        txt_labels : list or dict
            List of labels (or dict of {int : string} pairs) to be assigned to nodes
        w_adapt : 2D np.array
            Adaptation weights matrix
        w_combine : 2D np.array
            Combination weights matrix
        random_seed : int, optional
            Random seed for the graph generator
        G : networkx.Graph, np.array
            A custom graph or an adjacency matrix
        """
        if G is None:
            while True:
                G = nx.expected_degree_graph(
                    w=[avg_deg for _ in range(nodes)], seed=random_seed, selfloops=False
                )
                if nx.is_connected(G):
                    break
            self.nnodes = G.order()
            self._adj_mat = nx.to_numpy_array(
                G, nodelist=sorted(G.nodes), dtype=np.int
            ) + np.eye(self.nnodes, dtype=np.int)
        else:
            try:
                # networkx Graph
                self.nnodes = G.order()
                self._adj_mat = nx.to_numpy_array(
                    G, nodelist=sorted(G.nodes), dtype=np.int
                ) + np.eye(self.nnodes, dtype=np.int)
            except AttributeError:
                # Adjacency matrix
                self._adj_mat = np.asarray(G) + np.eye(
                    np.asarray(G).shape[0], dtype=np.int
                )
                self.nnodes = self._adj_mat.shape[0]
            except:
                print(
                    f"G must be a networkX.Graph object or an ajacency matrix, got {type(G)}."
                )
                raise

        self.kfs = [None] * self.nnodes
        self._txt_labels = [None] * self.nnodes

        self._weights_c = [None] * self.nnodes
        self._nbh_est = [None] * self.nnodes

        # Initialize adaptation and combination weight matrices
        self.w_adapt = np.ones_like(self._adj_mat) if w_adapt is None else w_adapt
        self.w_combine = np.ones_like(self._adj_mat) if w_combine is None else w_combine

        self.assign(init, txt_labels)

    def assign(self, init=None, txt_labels=None):
        """ Assign KalmanFilter objects to nodes.

        Parameters
        ----------
        init : list or dict
            List of KalmanFilter objects (or dict of {int : KalmanFilter} pairs)
            to be assigned to nodes
        txt_labels : list or dict
            List of strings (or dict of {int : string} pairs) to be assigned to nodes
        """
        # TODO Check if inputs are KF objects?
        if init is not None:
            try:
                if len(init) > self.nnodes:
                    print("[KFNet] Warning: size of init is more than number of nodes")
                # Dict initilization
                for idx, kf in init.items():
                    try:
                        self.kfs[idx] = kf
                    except IndexError:
                        continue
            except AttributeError:
                # List initialization
                n = self.nnodes if self.nnodes <= len(init) else len(init)
                self.kfs[:n] = init[:n]
            except KeyError:
                pass
            except:
                raise TypeError(f"init must be a list or a dict, got {type(init)}.")

        if txt_labels is not None:
            try:
                if len(txt_labels) > self.nnodes:
                    print(
                        "[KFNet] Warning: size of txt_labels is more than number of nodes"
                    )
                # Dict initilization
                for idx, lbl in txt_labels.items():
                    try:
                        self._txt_labels[idx] = lbl
                    except IndexError:
                        continue
            except AttributeError:
                # List initialization
                n = self.nnodes if self.nnodes <= len(txt_labels) else len(txt_labels)
                self._txt_labels[:n] = txt_labels[:n]
            except KeyError:
                pass  # Some node n not in G could be fine
            except:
                raise TypeError(
                    f"txt_labels must be a list or a dict, got {type(txt_labels)}."
                )

    def generate_txt_labels(self):
        """ Generate text labels for nodes based on their motion model.
        """
        rwm_cnt = 0
        cvm_cnt = 0
        cam_cnt = 0

        for i, kf in enumerate(self.kfs):
            try:
                model = kf.model
            except AttributeError:
                continue
            if isinstance(model, RWModel):
                rwm_cnt += 1
                txt_label = f"RWM_{rwm_cnt}"
            elif isinstance(model, CVModel):
                cvm_cnt += 1
                txt_label = f"CVM_{cvm_cnt}"
            elif isinstance(model, CAModel):
                cam_cnt += 1
                txt_label = f"CAM_{cam_cnt}"
            else:
                txt_label = "N/A"

            self._txt_labels[i] = txt_label

    def __getitem__(self, key):
        return self.kfs[key]

    def __setitem__(self, key, val):
        try:
            # val = (kf, label)
            self.kfs[key] = val[0]
            self._txt_labels[key] = val[1]
        except:
            # val = kf
            self.kfs[key] = val

    def __iter__(self):
        # Iterate through assigned nodes
        return (kf for kf in self.kfs if kf is not None)

    def draw_network(self, node_size=1000, figsize=(10, 7)):
        # TODO Distinct colors for RWM/CVM/CAM?
        """ Draw network.

        Parameters
        ----------
        node_size: int, default 1000
            Node size
        figsize: tuple (int, int), default (10, 7)
            Figure size (for pyplot)
        """
        G = nx.from_numpy_array(self._adj_mat - np.eye(self.nnodes, dtype=np.int))

        in_nodes = []
        out_nodes = []
        node_labels = {}
        for i, kf in enumerate(self.kfs):
            if kf is not None:
                in_nodes.append(i)
            else:
                out_nodes.append(i)

            node_lbl = str(i)
            if self._txt_labels[i] is not None:
                node_lbl += ":" + self._txt_labels[i]
            node_labels[i] = node_lbl

        plt.figure(figsize=figsize)
        # pos = nx.circular_layout(G)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=in_nodes,
            node_size=node_size,
            node_color="c",
            label="Initialized",
        )
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=out_nodes,
            node_size=node_size,
            node_color="r",
            label="Uninitialized",
        )
        nx.draw_networkx_labels(
            G, pos=pos, labels=node_labels, font_size=12, font_weight="bold",
        )
        nx.draw_networkx_edges(G, pos=pos, alpha=0.6)

        plt.axis("off")
        plt.tight_layout()
        plt.legend(scatterpoints=1)
        plt.show()

    def predict(self, u=None):
        """ Run KalmanFilter predict step on all nodes.

        Parameters
        ----------
        u : np.array, optional
            Optional control vector
        """
        if self._is_fully_init():
            for kf in self.kfs:
                kf.predict(u=u)
        else:
            self._print_uninitialized()

    def update(self, y):
        """ Run KalmanFilter update step on all nodes.

        Parameters
        ----------
        y : np.array
            Observation for this time step
        """
        if self._is_fully_init():
            # TODO Missing observations?
            if y.ndim == 2:
                # Different observation for each node
                for yi, i in zip(y, range(self.nnodes)):
                    self.kfs[i].update(y=yi, w=self.w_adapt[i, i])
            else:
                # The same observation for all nodes
                for i in range(self.nnodes):
                    self.kfs[i].update(y=y, w=self.w_adapt[i, i])
        else:
            self._print_uninitialized()

    def adapt(self):
        """ Adaptation phase - incorporates observations from neighbor nodes
        using update().
        """
        if self._is_fully_init():
            for i in range(self.nnodes):
                kf = self.kfs[i]

                # Get neighbor node indices
                # Exclude current node from nbh
                # Get references to KalmanFilter objects
                # Get neighborhood observations
                # Get adaptation phase weights
                nbh_indices = self._adj_mat[i].nonzero()[0]
                nbh_indices = nbh_indices[nbh_indices != i]
                nbh = [self.kfs[j] for j in nbh_indices]
                nbh_obs = self._get_nbh_observations(nbh)
                weights_a = self.w_adapt[i, nbh_indices]

                y_tmp = kf.y
                for (yi, Ri), wi in zip(nbh_obs, weights_a):
                    # Technically, observation matrix H should be passed in as well
                    kf.update(y=yi, R=Ri, w=wi)

                # update() saves latest observation
                # restore node's original observation
                kf.y = y_tmp

        else:
            self._print_uninitialized()

    def combine(self):
        """ Combination phase - combines estimates from neighbors using
        covariance intersection algorithm.
        """
        if self._is_fully_init():
            # Get the most recent neighborhood estimates
            self._update_nbh_estimates(incl_self=True)

            # Combine estimates using covariance intersection
            for i in range(self.nnodes):
                kf = self.kfs[i]
                nbh_est = self._nbh_est[i]
                wi = self._weights_c[i]

                # Unzip into array of estimates and array of cov. matrices
                xi_arr, Pi_arr = zip(*nbh_est)
                ci_est = self._cov_intersect(xi_arr, Pi_arr, wi)
                kf.set_estimate(*ci_est)
        else:
            self._print_uninitialized()

    def log(self):
        for kf in self.kfs:
            kf._log()

    def time_step(
        self,
        y=None,
        predict=True,
        update=True,
        adapt=True,
        combine=True,
        reset_strategy="mean",
        reset_thresh=5.0,
        c=1.0,
    ):
        """ Runs one iteration of diffusion Kalman filtering
        using adapt-then-combine strategy.

        Parameters
        ----------
        y : np.array of shape (n, m)
            n - number of nodes
            m - observation dimension

            Set of observations for this time step, single observations
            can be used as well (identical observation for all nodes)
        predict : bool, default True
            Run predict step
        update : bool, default True
            Run update step
        adapt : bool, default True
            Run adapt step
        combine : bool, default True
            Run combine step
        reset_strategy : [None, "mean", "ci"]
            "mean": centroid is calculated as the uniformly weighted
                arithmetic mean of neighbor estimates
            "ci": centroid is calculated as covariance intersection
                of neighbor estimates (i. e., "uncertainty weighted")
        reset_threshold : float, default 5.0
            Maximum accepted distance from the centroid before the filter reset
        c : float, default 1.0
            TODO
        """
        if self._is_fully_init():
            if predict:
                self.predict()
            if reset_strategy is not None:
                self.reset_filters(reset_strategy, reset_thresh, c)
            if y is not None:
                if update:
                    self.update(y)
                # if reset_strategy is not None:
                #     self.reset_filters(reset_strategy, reset_thresh, c)
                if update and adapt:
                    self.adapt()
            if combine:
                # if reset_strategy is not None:
                #     self.reset_filters(reset_strategy, reset_thresh, c)
                self.combine()
            self.log()
        else:
            self._print_uninitialized()

    def _is_fully_init(self):
        return not None in self.kfs

    def _print_uninitialized(self):
        uninitialized = [n for n in range(self.nnodes) if self.kfs[n] is None]
        raise RuntimeError(f"Nodes {uninitialized} are uninitialized")

    def observation_covs(self):
        """
        Return observation noise covariance matrices.

        Returns
        -------
        Array of observation noise covariance matrices.
        """
        return np.array([kf.model.R for kf in self.kfs])

    def reset_filters(self, reset_strategy, reset_thresh, c=1.0):
        """

        Parameters
        ----------
        reset_strategy : [None, "mean", "ci"]
            "mean": centroid is calculated as the uniformly weighted
                arithmetic mean of neighbor estimates
            "ci": centroid is calculated as covariance intersection
                of neighbor estimates (i. e., "uncertainty weighted")
        reset_threshold : float
            Maximum accepted distance from the centroid before the filter reset
        c : float, default 1.0

        """
        if self._is_fully_init():
            if reset_strategy is not None:
                # Get the most recent neighborhood estimates
                self._update_nbh_estimates(incl_self=False)
                for i in range(self.nnodes):
                    self._check_reset_cond(
                        self.kfs[i],
                        self._nbh_est[i],
                        self._weights_c[i],
                        reset_strategy,
                        reset_thresh,
                        c,
                    )
        else:
            self._print_uninitialized()

    @staticmethod
    def _check_reset_cond(kf, nbh_est, nbh_w, reset_strategy, reset_thresh, c):
        if not isinstance(reset_thresh, (float, int)):
            raise TypeError(f"Number expected, got {type(reset_thresh)}")

        if len(nbh_est) < 2:
            return

        if reset_strategy == "mean":
            # Centroid as arithmetic mean of estiamtes
            # TODO weights?
            ctr = np.asarray([x for (x, _) in nbh_est]).mean(axis=0)
        elif reset_strategy == "ci":
            # Centroid calculated using covariance intersection
            xi_arr, Pi_arr = zip(*nbh_est)
            ctr, _ = KFNet._cov_intersect(xi_arr, Pi_arr, nbh_w)
        else:
            raise ValueError(f"Invalid reset_strategy: {reset_strategy}")

        dist_x_ctr = np.linalg.norm(ctr - kf.x)

        obs, _ = kf.get_observation()
        try:
            ndim_obs = obs.shape[0]
            dist_ctr_obs = np.linalg.norm(obs - ctr[:ndim_obs])
        except (TypeError, AttributeError):
            # No observation is available
            pass

        if dist_x_ctr >= reset_thresh:
            try:
                if dist_ctr_obs >= reset_thresh:
                    xnew = obs
                else:
                    xnew = ctr
            except NameError:
                # No observation is available
                xnew = ctr

            # P = c^(# of resets) * P0
            Pnew = (c ** len(kf._reset_log)) * kf._P0
            kf.reset_filter(xnew, Pnew)

    def _update_nbh_estimates(self, incl_self=True):
        if self._is_fully_init():
            for i in range(self.nnodes):
                kf = self.kfs[i]
                nbh_indices = self._adj_mat[i].nonzero()[0]
                if incl_self is False:
                    # Don't include self for resets
                    nbh_indices = nbh_indices[nbh_indices != i]
                nbh = [self.kfs[j] for j in nbh_indices]
                wi = self.w_combine[i, nbh_indices]

                # Only use estimates from models of the same complexity or better
                # Also need to obtain the correct weights
                self._nbh_est[i], self._weights_c[i] = self._get_nbh_estimates(
                    kf, nbh, wi
                )

    @staticmethod
    def _get_nbh_estimates(kf, nbh, nbh_w, indices=None):
        """ Get estimates from agents in neighborhood. This method should be
        called after predict() and update() but before combine().

        Parameters
        ----------
        kf : KalmanFilter object
            KalmanFilter estimator
        nbh :

        indices : list-of-lists/arrays, optional
            A list, the size of neighborhood, of indices of variables over
            which to get marginal distributions for each agent in neighborhood

            Selects ndim first variables by default, ndim is the dimension of
            this agent's estimate
        """
        if not indices:
            # Default case
            indices = [None for i in range(len(nbh))]

        nbh_est = []
        weights = []
        for nbh, i, w in zip(nbh, indices, nbh_w):
            # Only consider estimates from models with the same/higher complexity
            if nbh._ndim >= kf._ndim:
                # i is usually None -> select first _ndim variables
                nbh_est.append(
                    nbh.get_estimate(
                        indices=np.arange(kf._ndim, dtype=np.int) if not i else i
                    )
                )
                weights.append(w)

        return nbh_est, weights

    @staticmethod
    def _get_nbh_observations(nbh):
        """ Get latest observations from agents in neighborhood.

        Parameters
        ----------
        nbh : list
            List of references to KalmanFilter objects

        Returns
        -------
            List of observations from nbh
        """
        return [kf.get_observation() for kf in nbh]

    @staticmethod
    def _cov_intersect(xi_arr, Pi_arr, weights=None, normalize=True):
        """ Combine estimates using covariance intersection.

        Parameters
        ----------
        xi_arr : np.array
            Array of state estimates
        Pi_arr : np.array
            Array of corresponding covariance matrices
        weights : np.array
            Array of weights
        normalize : bool, optional, default True
            Normalize weights

        Returns
        -------
        (xnew, P_new): (np.array, np.array)
            New estimate obtained from covariance intersection
            of neighborhood estimates
        """
        if weights is None:
            weights = np.ones(shape=len(xi_arr))
            weights /= np.sum(weights)

        if normalize is True:
            weights = np.asarray(weights) / np.sum(weights)

        # Loop based version
        # xnew = np.zeros_like(xi_arr[0])
        # P_newinv = np.zeros_like(Pi_arr[0])

        # for w, x, P in zip(weights, xi_arr, Pi_arr):
        #     Pinv = np.linalg.inv(P)
        #     P_newinv += w * Pinv
        #     xnew += w * Pinv.dot(x.T)

        # P_new = np.linalg.inv(P_newinv)
        # xnew = P_new.dot(xnew)

        # NumPy vectorized version
        # xis, Pis = zip(*nbh_est)

        P_newinv = weights[:, np.newaxis, np.newaxis] * np.linalg.inv(Pi_arr)
        xnew = np.sum(np.einsum("ikj, ij -> ik", P_newinv, xi_arr), axis=0)

        P_new = np.linalg.inv(np.sum(P_newinv, axis=0))
        xnew = P_new.dot(xnew)

        return (xnew, P_new)

    @property
    def w_adapt(self):
        """ Weight matrix for adapt step.
        """
        return self._w_adapt

    @w_adapt.setter
    def w_adapt(self, w_adapt):
        """ Set weight matrix for adapt step.
        """
        if (w_adapt is not None) and (
            np.asarray(w_adapt).shape != (self.nnodes, self.nnodes)
        ):
            raise ValueError(
                f"Weight matrix of shape {self.nnodes, self.nnodes} expected."
            )
        self._w_adapt = w_adapt

    @property
    def w_combine(self):
        """ Weight matrix for combine step.
        """
        return self._w_combine

    @w_combine.setter
    def w_combine(self, w_combine):
        """ Set weight matrix for combine step.
        """
        if (w_combine is not None) and (
            np.asarray(w_combine).shape != (self.nnodes, self.nnodes)
        ):
            raise ValueError(
                f"Weight matrix of shape {self.nnodes, self.nnodes} expected."
            )
        self._w_combine = w_combine

    @property
    def adj_mat(self):
        return self._adj_mat - np.eye(self.nnodes)

    @adj_mat.setter
    def adj_mat(self, data):
        self.nnodes = data.shape[0]
        self._adj_mat = data + np.eye(data.shape[0], dtype=np.int)
