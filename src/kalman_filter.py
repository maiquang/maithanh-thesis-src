import numpy as np
from .statespace import StateSpace

class KalmanFilter():
    def __init__(self, model):
        if not isinstance(model, StateSpace):
            raise TypeError(f"StateSpace object expected, got {type(model)}")

    def predict(self):
        pass

    def update(self):
        pass

    def cov_intersect(self):
        pass

    def log(self):
        pass
