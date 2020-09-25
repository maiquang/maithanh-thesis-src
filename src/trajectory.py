import numpy as np
from scipy.stats import multivariate_normal as mvn

class trajectory():
	
	def __init__(self, A, H, Q, R, init_state, B=None, u=None, random_state=0, n=100):
		# Simulation parameters
		self.n = n
		self.seed = random_state
		
		# Process model 
		self.A = A
		self.B = B
		self.u = u
		self.Q = Q
		self.x0 = init_state
		
		# Measurement model
		self.H = H
		self.R = R
		
		# Store actual state & measurements
		self.X = np.zeros(shape=(self.A.shape[0], self.n))
		self.Y = np.zeros(shape=(self.H.shape[0], self.n))
		
		# Simulate trajectory
		self._simulate()
		
	
	def _simulate(self):
		np.random.seed(self.seed)
		
		x = self.x0
		for t in range(self.n):
			x = self.A @ x + mvn.rvs(cov=self.Q)
			if self.B is not None and self.u is not None:
				x += self.B @ u
			
			y = self.H @ x + mvn.rvs(cov=self.R)
			
			self.X[:, t] = x.flatten()
			self.Y[:, t] = y.flatten()