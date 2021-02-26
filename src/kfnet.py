import copy
import networkx as nx
from matplotlib import pyplot as plt

# from kalmanfilter import KalmanFilter
# from statespace import *


class KFNet:
    # kfn = KFNet(nodes, deg, dict/list, seed)
    # kfn.plot_network(labels="txt/[idx]")
    # kfn.assign_nodes(dict/list)

    def __init__(
        self, nodes=15, avg_deg=10, init=None, txt_labels=None, random_seed=None, G=None
    ):
        if G is None:
            self.G = nx.expected_degree_graph(
                w=[avg_deg for _ in range(nodes)], seed=random_seed, selfloops=False
            )
        else:
            if not isinstance(G, nx.Graph):
                raise TypeError(f"G must be a networkX.Graph object, got {type(G)}.")
            self.G = copy.deepcopy(G)

        # degrees = [d for g, d in self.G.degree]
        # print(sum(degrees) / nodes)

        if init is not None:
            if isinstance(init, dict):
                if len(init) > self.G.order():
                    print("[KFNet] Warning: size of init is more than number of nodes")
                # dict init
                for idx, kf in init.items():
                    self.G.nodes[idx]["kf"] = kf
            elif isinstance(init, list):
                if len(init) > self.G.order():
                    print("[KFNet] Warning: size of init is more than number of nodes")
                # list init
                n = self.G.order() if self.G.order() <= len(init) else len(init)
                for idx in range(n):
                    self.G.nodes[idx]["kf"] = init[idx]

            else:
                raise TypeError(f"init must be a list or a dict, got {type(init)}.")

        print(self.G.nodes.data())
        # txt-labels init

        # TODO init using dict {int: model}
        # TODO init using list/array
        # TODO init neighborhood -> separate update method?
        # TODO init txt-labels as node attributes
        # TODO access using []
        # TODO batch init using dict/list
        # TODO maybe custom topologies constructed using networkx

    def draw_network(self, labels="txt"):
        nx.draw_circular(self.G, with_labels=True)

        # TODO draw with idx labels
        # TODO draw with txt labels
        # TODO draw with
        # TODO assigned nodes with different color

        plt.show()

    # kfn.predict()
    # kfn.update()
    # kfn.adapt()
    # kfn.combine() -> kfn.covariance_intersection() ?
    # kfn.time_step()
    # kfn.estimates ??

    def predict(self):
        pass

    def update(self):
        pass

    def adapt(self):
        pass

    def combine(self):
        pass

    def time_step(self):
        pass
