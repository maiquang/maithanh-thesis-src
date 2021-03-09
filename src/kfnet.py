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
        self, nodes=15, avg_deg=10, init=None, txt_labels=None, random_seed=None, G=None
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

    # kfn.estimates ??

    def _init_nbhood(self):
        for node in self.G:
            for neighbor in self.G.neighbors(node):
                # "self" was added in initialization stage
                self.G.nodes[node]["nbhood"].append(self.G.nodes[neighbor]["kf"])

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
                for yi, (_, kf) in zip(y, self.G.nodes(data="kf")):
                    kf.update(y=yi)
            else:
                for _, kf in self.G.nodes(data="kf"):
                    kf.update(y=y)
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

                y_tmp = kf.y
                for yi in nbh_obs:
                    kf.update(y=yi)

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
                nbh_est = attrs["nbh_est"] = self._get_nbh_estimates(kf, nbhood)
                if reset_strategy is not None:
                    self._check_reset_cond(kf, nbh_est, reset_strategy, reset_thresh)

            # Combine estimates using covariance intersection
            for _, attrs in self.G.nodes(data=True):
                kf = attrs["kf"]
                nbh_est = attrs["nbh_est"]

                # Covariance intersection
                # Possibly other strategies?
                ci_est = self._cov_intersect(kf, nbh_est)
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

    @staticmethod
    def _check_reset_cond(kf, nbh_est, reset_strategy, reset_thresh):
        if not isinstance(reset_thresh, (float, int)):
            raise TypeError(f"Number expected, got {type(reset_thresh)}")

        if len(nbh_est) < 3:
            return

        if reset_strategy == "mean":
            ctr = np.asarray([x for (x, P) in nbh_est[1:]]).mean(axis=0)
        elif reset_strategy == "ci":
            ctr, _ = KFNet._cov_intersect(kf, nbh_est[1:])
        else:
            raise ValueError(f"Invalid reset_strategy: {reset_strategy}")

        dist_x_ctr = np.linalg.norm(ctr - kf.x)

        obs = kf.get_observation()
        try:
            dist_ctr_obs = np.linalg.norm(obs - ctr)
        except:
            # Observation is None
            return

        if dist_x_ctr >= reset_thresh:
            if dist_ctr_obs >= reset_thresh:
                xnew = obs
            else:
                xnew = ctr
            kf.reset_filter(xnew)
            nbh_est[0] = kf.get_estimate()

    @staticmethod
    def _get_nbh_estimates(kf, nbhood, indices=None):
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
        for nbh, i in zip(nbhood, indices):
            # Only consider estimates from models with same/higher complexity
            if nbh._ndim >= kf._ndim:
                # i is usually None -> select first _ndim variables
                nbh_ests.append(
                    nbh.get_estimate(
                        indices=np.arange(kf._ndim, dtype=np.int) if not i else i
                    )
                )

        return nbh_ests

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
    def _cov_intersect(kf, nbh_est, weights=None, normalize=True):
        """ Calculate combined estimate from neighborhood agents' estimates
        using the covariance intersection algorithm. Updated neighborhood
        estimates should be obtained by calling _get_nbh_estimates()
        before each _cov_intersect() call.

        Parameters
        ----------
        kf : KalmanFilter object
            KalmanFilter estimator
        nbh_est : list/array of (x, P) tuples
            Neighborhood estimates
        weights : list-like
            Weights for each agent in the neighborhood
        normalize : bool, optional, default True
            Normalize weights

        Returns
        -------
        (xnew, P_new): (np.array, np.array)
            New estimate obtained from covariance intersection
            of neighborhood estimates
        """
        if weights is None:
            weights = np.ones(shape=len(nbh_est))
            weights /= np.sum(weights)

        if normalize is True:
            weights = np.asarray(weights) / np.sum(weights)

        P_newinv = np.zeros_like(kf.P)
        xnew = np.zeros_like(kf.x)

        for w, (x, P) in zip(weights, nbh_est):
            Pinv = np.linalg.inv(P)
            P_newinv += w * Pinv
            xnew += w * Pinv.dot(x.T)

        P_new = np.linalg.inv(P_newinv)
        xnew = P_new.dot(xnew)

        return (xnew, P_new)
