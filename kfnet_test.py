from src.statespace import RWModel, CVModel, CAModel
from src.kalmanfilter import KalmanFilter
from src.kfnet import KFNet
import numpy as np

import networkx as nx

# RANDOM_SEED = None
# ndat = 100
q = 9.5e-5  # Process noise intensity
r = 1.3  # Observation noise std
c_rwm = 10  # RWM process noise multiplicative constant
expf = 0.95  # Exponential forgetting param
# reset_thresh = 3.0  # Filter reset threshold (Euclidean distance from centroid)
# init_state = np.zeros(6)

if __name__ == "__main__":
    kfs = [
        KalmanFilter(RWModel(c_rwm * q, r), lambda_expf=expf),
        KalmanFilter(CVModel(q, r)),
        KalmanFilter(CAModel(q, r)),
        KalmanFilter(CAModel(q, r)),
        KalmanFilter(CAModel(q, r)),
        KalmanFilter(CAModel(q, r)),
    ]

    # kfs_dict = {idx: kf for idx, kf in enumerate(kfs)}
    # labels = ["RWM_1", "CVM_1", "CAM_1"]

    G = np.eye(6)
    kfn = KFNet(6, 3, G=G, init=kfs)

    print(kfn._adj_mat)
    kfn.generate_txt_labels()
    kfn.draw_network()
