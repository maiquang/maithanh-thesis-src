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
    G : networkx.Graph
        A custom networkx.Graph to be used

    Attributes
    ----------
    G : networkx.Graph
        Underlying network.G graph represeting the network topology

    Methods
    -------
    assign(init=None, txt_labels=None)
        Assign KalmanFilter objects to nodes.
    generate_txt_labels()
        Generate text labels for nodes based on their motion model.
    draw_network(self, node_size, figsize)
        Draw network.
    predict()
        Run KalmanFilter predict step on all nodes.
    update(y)
        Run KalmanFilter update step on all nodes.
    adapt()
        Adaptation step - incorporates observations from neighbor nodes
        using update().
    combine(reset_strategy, reset_thresh)
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
        G : networkx.Graph
            A custom networkx.Graph to be used
        """
        if G is None:
            while True:
                self.G = nx.expected_degree_graph(
                    w=[avg_deg for _ in range(nodes)], seed=random_seed, selfloops=False
                )
                if nx.is_connected(self.G):
                    break
        else:
            if not isinstance(G, nx.Graph):
                raise TypeError(f"G must be a networkX.Graph object, got {type(G)}.")
            self.G = copy.deepcopy(G)

        # Keep track of initialized nodes
        self._node_set = set()

        self.w_adapt = w_adapt
        self.w_combine = w_combine

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
                if len(init) > self.G.order():
                    print("[KFNet] Warning: size of init is more than number of nodes")
                # Dict initilization
                for idx, kf in init.items():
                    self.G.nodes[idx]["kf"] = kf
                    self.G.nodes[idx]["nbhood"] = [kf]
                self._node_set.update(init.keys())
            except AttributeError:
                # List initialization
                n = self.G.order() if self.G.order() <= len(init) else len(init)
                for idx in range(n):
                    self.G.nodes[idx]["kf"] = init[idx]
                    self.G.nodes[idx]["nbhood"] = [init[idx]]
                self._node_set.update(range(n))
            except KeyError:
                pass  # Some node n not in G could be fine
            except:
                raise TypeError(f"init must be a list or a dict, got {type(init)}.")

        if txt_labels is not None:
            try:
                if len(txt_labels) > self.G.order():
                    print(
                        "[KFNet] Warning: size of txt_labels is more than number of nodes"
                    )
                # Dict initilization
                for idx, kf in txt_labels.items():
                    self.G.nodes[idx]["txt_label"] = kf
            except AttributeError:
                # List initialization
                n = (
                    self.G.order()
                    if self.G.order() <= len(txt_labels)
                    else len(txt_labels)
                )
                for idx in range(n):
                    self.G.nodes[idx]["txt_label"] = txt_labels[idx]
            except:
                raise TypeError(
                    f"txt_labels must be a list or a dict, got {type(txt_labels)}."
                )

        if self._is_fully_init():
            self._init_nbhood()

    def generate_txt_labels(self):
        """ Generate text labels for nodes based on their motion model.
        """
        rwm_cnt = 0
        cvm_cnt = 0
        cam_cnt = 0

        for _, attrs in self.G.nodes(data=True):
            try:
                model = attrs["kf"].model
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

                attrs["txt_label"] = txt_label
            except KeyError:
                pass

    def __getitem__(self, key):
        return self.G.nodes[key]["kf"]

    def __setitem__(self, key, val):
        try:
            # val = (kf, label)
            self.G.nodes[key]["kf"] = val[0]
            self.G.nodes[key]["nbhood"] = val[0]
            self.G.nodes[key]["txt_label"] = val[1]
        except:
            # val = kf
            self.G.nodes[key]["kf"] = val
            self.G.nodes[key]["nbhood"] = val
        self._node_set.add(key)

        if self._is_fully_init():
            self._init_nbhood()

    def __iter__(self):
        # Iterate through assigned nodes
        return (kf for _, kf in self.G.nodes(data="kf") if kf is not None)

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
        in_nodes = []
        out_nodes = []
        node_labels = {}
        for node, attrs in self.G.nodes(data=True):
            if ("kf" in attrs) and (attrs["kf"] is not None):
                in_nodes.append(node)
            else:
                out_nodes.append(node)

            node_lbl = str(node)
            if ("txt_label" in attrs) and (attrs["txt_label"] is not None):
                node_lbl += ":" + attrs["txt_label"]
            node_labels[node] = node_lbl

        # pos = nx.circular_layout(self.G)
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            nodelist=in_nodes,
            node_size=node_size,
            node_color="c",
            label="Initialized",
        )
        nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            nodelist=out_nodes,
            node_size=node_size,
            node_color="r",
            label="Uninitialized",
        )
        nx.draw_networkx_labels(
            self.G, pos=pos, labels=node_labels, font_size=12, font_weight="bold",
        )
        nx.draw_networkx_edges(self.G, pos=pos, alpha=0.6)

        plt.axis("off")
        plt.tight_layout()
        plt.legend(scatterpoints=1)
        plt.show()

    def _init_nbhood(self):
        # Default weights, if they are not provided
        if self.w_adapt is None:
            self.w_adapt = np.ones((self.G.order(), self.G.order()))

        if self.w_combine is None:
            self.w_combine = np.ones((self.G.order(), self.G.order()))

        for node, attrs in self.G.nodes(data=True):
            w_adapt = [self.w_adapt[node, node]]
            w_combine = [self.w_combine[node, node]]
            for neighbor in self.G.neighbors(node):
                # "self" was added in initialization stage
                attrs["nbhood"].append(self.G.nodes[neighbor]["kf"])
                # Init weights from matrices
                w_adapt.append(self.w_adapt[node, neighbor])
                w_combine.append(self.w_combine[node, neighbor])
            attrs["w_adapt"] = w_adapt
            attrs["w_combine"] = w_combine

    def predict(self, u=None):
        """ Run KalmanFilter predict step on all nodes.

        Parameters
        ----------
        u : np.array, optional
            Optional control vector
        """
        if self._is_fully_init():
            for _, kf in self.G.nodes(data="kf"):
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
                # The same observation for all nodes
                for yi, (node, kf) in zip(y, self.G.nodes(data="kf")):
                    kf.update(y=yi, w=self.w_adapt[node, node])
            else:
                # Different observation for each node
                for node, kf in self.G.nodes(data="kf"):
                    kf.update(y=y, w=self.w_adapt[node, node])
        else:
            self._print_uninitialized()

    def adapt(self):
        """ Adaptation step - incorporates observations from neighbor nodes
        using update().
        """
        if self._is_fully_init():
            for _, attrs in self.G.nodes(data=True):
                kf = attrs["kf"]
                nbhood = attrs["nbhood"]
                nbh_obs = attrs["nbh_obs"] = self._get_nbh_observations(nbhood[1:])
                w_adapt = attrs["w_adapt"][1:]

                y_tmp = kf.y
                for (yi, Ri), w in zip(nbh_obs, w_adapt):
                    # Technically, observation matrix H should be passed in as well
                    kf.update(y=yi, R=Ri, w=w)

                # update() saves latest observation
                # restore node's original observation
                kf.y = y_tmp

        else:
            self._print_uninitialized()

    def combine(self, reset_strategy="mean", reset_thresh=None):
        """ Combination step - combines estimates from neighbors using
        covariance intersection algorithm.

        Parameters
        ----------
        reset_strategy : [None, "mean", "ci"]
            "mean": centroid is calculated as the uniformly weighted
                arithmetic mean of neighbor estimates
            "ci": centroid is calculated as covariance intersection
                of neighbor estimates (i. e., "uncertainty weighted")
        reset_threshold : float
            Maximum accepted distance from the centroid before the filter reset
        """
        if self._is_fully_init():
            # Get neighborhood estimates
            for _, attrs in self.G.nodes(data=True):
                kf = attrs["kf"]
                nbhood = attrs["nbhood"]
                nbh_w = attrs["w_combine"]

                # Only use estimates from models of the same complexity or better
                # Also need to obtain the correct weights
                attrs["nbh_est"], attrs["w_combine"] = self._get_nbh_estimates(
                    kf, nbhood, nbh_w
                )
                if reset_strategy is not None:
                    self._check_reset_cond(
                        kf,
                        attrs["nbh_est"],
                        attrs["w_combine"],
                        reset_strategy,
                        reset_thresh,
                    )

            # Combine estimates using covariance intersection
            for _, attrs in self.G.nodes(data=True):
                kf = attrs["kf"]
                nbh_est = attrs["nbh_est"]
                w_combine = attrs["w_combine"]

                # Unzip into array of estimates and array of cov. matrices
                xi_arr, Pi_arr = zip(*nbh_est)
                ci_est = self._cov_intersect(xi_arr, Pi_arr, w_combine)
                kf.set_estimate(*ci_est)
        else:
            self._print_uninitialized()

    def log(self):
        for _, kf in self.G.nodes(data="kf"):
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
        reset_threshold : float
            Maximum accepted distance from the centroid before the filter reset
        """
        if self._is_fully_init():
            if predict:
                self.predict()
            if y is not None:
                if update:
                    self.update(y)
                if update and adapt:
                    self.adapt()
            if combine:
                self.combine(reset_strategy, reset_thresh)
            self.log()
        else:
            self._print_uninitialized()

    def _is_fully_init(self):
        return len(self._node_set) == self.G.order()

    def _print_uninitialized(self):
        uninitialized = [n for n, d in self.G.nodes(data="kf") if d is None]
        raise RuntimeError(f"Nodes {uninitialized} are uninitialized")

    def reset_filters(self, reset_strategy, reset_thresh):
        """
        """
        if self._is_fully_init():
            # Get neighborhood estimates
            for _, attrs in self.G.nodes(data=True):
                kf = attrs["kf"]
                nbhood = attrs["nbhood"]
                nbh_w = attrs["w_combine"]

                # Only use estimates from models of the same complexity or better
                # Also need to obtain the correct weights
                attrs["nbh_est"], attrs["w_combine"] = self._get_nbh_estimates(
                    kf, nbhood, nbh_w
                )
                if reset_strategy is not None:
                    self._check_reset_cond(
                        kf,
                        attrs["nbh_est"],
                        attrs["w_combine"],
                        reset_strategy,
                        reset_thresh,
                    )
        else:
            self._print_uninitialized()

    @staticmethod
    def _check_reset_cond(kf, nbh_est, nbh_w, reset_strategy, reset_thresh):
        if not isinstance(reset_thresh, (float, int)):
            raise TypeError(f"Number expected, got {type(reset_thresh)}")

        if len(nbh_est) < 3:
            return

        if reset_strategy == "mean":
            # Centoid as arithmetic mean of estiamtes
            # TODO weights?
            ctr = np.asarray([x for (x, P) in nbh_est[1:]]).mean(axis=0)
        elif reset_strategy == "ci":
            # Centoid calculated using covariance intersection
            xi_arr, Pi_arr = zip(*nbh_est[1:])
            ctr, _ = KFNet._cov_intersect(xi_arr, Pi_arr, nbh_w[1:])
        else:
            raise ValueError(f"Invalid reset_strategy: {reset_strategy}")

        dist_x_ctr = np.linalg.norm(ctr - kf.x)

        obs, _ = kf.get_observation()
        try:
            nbdim_obs = obs.shape[0]
            dist_ctr_obs = np.linalg.norm(obs - ctr[:nbdim_obs])
        except TypeError:
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

            # Update estimate in nbh_est array
            kf.reset_filter(xnew)
            nbh_est[0] = kf.get_estimate()

    @staticmethod
    def _get_nbh_estimates(kf, nbhood, nbh_w, indices=None):
        """ Get estimates from agents in neighborhood. This method should be
        called after predict() and update() but before combine().

        Parameters
        ----------
        kf : KalmanFilter object
            KalmanFilter estimator
        nbhood :

        indices : list-of-lists/arrays, optional
            A list, the size of neighborhood, of indices of variables over
            which to get marginal distributions for each agent in neighborhood

            Selects ndim first variables by default, ndim is the dimension of
            this agent's estimate
        """
        if not indices:
            # Default case
            indices = [None for i in range(len(nbhood))]

        nbh_ests = []
        weights = []
        for nbh, i, w in zip(nbhood, indices, nbh_w):
            # Only consider estimates from models with the same/higher complexity
            if nbh._ndim >= kf._ndim:
                # i is usually None -> select first _ndim variables
                nbh_ests.append(
                    nbh.get_estimate(
                        indices=np.arange(kf._ndim, dtype=np.int) if not i else i
                    )
                )
                weights.append(w)

        return nbh_ests, weights

    @staticmethod
    def _get_nbh_observations(nbhood):
        """ Get latest observations from agents in neighborhood.

        Parameters
        ----------
        nbhood : list
            List of references to KalmanFilter objects

        Returns
        -------
            List of observations from nbhood
        """
        return [kf.get_observation() for kf in nbhood]

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
            np.asarray(w_adapt).shape != (self.G.order(), self.G.order())
        ):
            raise ValueError(
                f"Weight matrix of shape {self.G.order(), self.G.order()} expected."
            )
        self._w_adapt = w_adapt

        if self._is_fully_init():
            self._init_nbhood()

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
            np.asarray(w_combine).shape != (self.G.order(), self.G.order())
        ):
            raise ValueError(
                f"Weight matrix of shape {self.G.order(), self.G.order()} expected."
            )
        self._w_combine = w_combine

        if self._is_fully_init():
            self._init_nbhood()

    def print_node_attr(self, attr, node="all"):
        if node == "all":
            for n in self.G.nodes(data=True if attr == "all" else attr):
                print(n)
        else:
            if attr == "all":
                print(self.G.nodes[node])
            else:
                print(self.G.nodes[node][attr])
