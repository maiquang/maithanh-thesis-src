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
        self.assign(init, txt_labels)
        print(self.G.nodes.data())
        # txt-labels init

        # TODO init using dict {int: model}
        # TODO init using list/array
        # TODO init neighborhood -> separate update method?
        # TODO init txt-labels as node attributes
        # TODO access using []
        # TODO batch init using dict/list
        # TODO maybe custom topologies constructed using networkx

    def assign(self, init=None, txt_labels=None):
        # TODO Check if inputs are KF objects?
        if init is not None:
            if isinstance(init, dict):
                if len(init) > self.G.order():
                    print("[KFNet] Warning: size of init is more than number of nodes")
                # Dict initilization
                for idx, kf in init.items():
                    self.G.nodes[idx]["kf"] = kf
            elif isinstance(init, list):
                if len(init) > self.G.order():
                    print("[KFNet] Warning: size of init is more than number of nodes")
                # List initialization
                n = self.G.order() if self.G.order() <= len(init) else len(init)
                for idx in range(n):
                    self.G.nodes[idx]["kf"] = init[idx]
            else:
                raise TypeError(f"init must be a list or a dict, got {type(init)}.")

        if txt_labels is not None:
            if isinstance(txt_labels, dict):
                if len(txt_labels) > self.G.order():
                    print(
                        "[KFNet] Warning: size of txt_labels is more than number of nodes"
                    )
                # Dict initilization
                for idx, label in txt_labels.items():
                    self.G.nodes[idx]["txt_label"] = label
            elif isinstance(txt_labels, list):
                if len(txt_labels) > self.G.order():
                    print(
                        "[KFNet] Warning: size of txt_labels is more than number of nodes"
                    )
                # List initialization
                n = (
                    self.G.order()
                    if self.G.order() <= len(txt_labels)
                    else len(txt_labels)
                )
                for idx in range(n):
                    self.G.nodes[idx]["txt_label"] = txt_labels[idx]
            else:
                raise TypeError(
                    f"init must be a list or a dict, got {type(txt_labels)}."
                )

    def draw_network(self, node_size=1000):
        pos = nx.circular_layout(self.G)

        in_nodes = []
        out_nodes = []
        node_labels = {}
        for node, attrs in self.G.nodes.data():
            if ("kf" in attrs) and (attrs["kf"] is not None):
                in_nodes.append(node)
            else:
                out_nodes.append(node)

            node_lbl = str(node)
            if ("txt_label" in attrs) and (attrs["txt_label"] is not None):
                node_lbl += ":" + attrs["txt_label"]
            node_labels[node] = node_lbl

        nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            nodelist=in_nodes,
            node_size=node_size,
            node_color="b",
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
        nx.draw_networkx_labels(self.G, pos=pos, labels=node_labels, font_size=12)
        nx.draw_networkx_edges(self.G, pos=pos, alpha=0.6)

        plt.axis("off")
        plt.tight_layout()
        plt.legend(scatterpoints=1)
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
