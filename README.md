# Distributed Kalman filtering under partially heterogeneous models (source code)

Source code for the simulation experiments of the master's thesis [**Distributed Kalman filtering under partially heterogeneous models (2021)**](https://raw.githubusercontent.com/maiquang/maithanh-thesis-src/master/DP_Mai_Thanh_Quang_2021.pdf) by Thanh Quang Mai, FIT CTU.

## Abstract
This thesis explores the problem of distributed Kalman filtering under partially heterogeneous models. A modification to the existing diffusion Kalman filter is proposed, enabling the employment of partially heterogeneous models in the diffusion networks. The performance of the less complex models is futher improved by the implementation of a node failure detection heuristic, resetting the failling nodes, and giving them a chance at a recovery.

# Summary
## Linear state-space model
Consider a model in the following form:
<!-- $$
\begin{aligned}
    x_t & = A_tx_{t-1} + B_tu_{t} + w_t, \\
    y_t & = H_tx_t + v_t,
\end{aligned}
$$ -->

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%0A%20%20%20%20x_t%20%26%20%3D%20A_tx_%7Bt-1%7D%20%2B%20B_tu_%7Bt%7D%20%2B%20w_t%2C%20%5C%5C%0A%20%20%20%20y_t%20%26%20%3D%20H_tx_t%20%2B%20v_t%2C%0A%5Cend%7Baligned%7D"></div>

where <!-- $x_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_t"> is the latent state, <!-- $y_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y_t"> is an observable (noisy) output, <!-- $u_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=u_t"> is a **known** input variable, and <!-- $A_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=A_t">, <!-- $B_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=B_t">, <!-- $H_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_t"> are known matrices of compatible dimensions. Noise variables <!-- $w_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=w_t"> and <!-- $v_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=v_t"> are zero-mean, mutually independent and identically distributed.

The goal is to estimate the latent state <!-- $x_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_t"> (e.g., an object's position in 2D space) based on inputs <!-- $u_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=u_t"> and noisy observations <!-- $y_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y_t"> (e.g., sensor readings) at each time-step <!-- $t = 0, 1, 2, \ldots$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=t%20%3D%200%2C%201%2C%202%2C%20%5Cldots">.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/83/Hmm_temporal_bayesian_net.svg">
</p>

## Kalman filter
The Kalman filter is used to estimate the latent state <!-- $x_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_t">. It does assume assume both the process noise <!-- $w_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=w_t"> and observation noise <!-- $v_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=v_t"> to be mutually independent and zero-mean Gaussian. The filtration runs in two steps:
- **prediction** - predicts the next state <!-- $x_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_t"> using the state evolution model above,
- **update** - incorporates a new measurement (observation) <!-- $y_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y_t"> into the estimate of the hidden state <!-- $x_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_t">.

## Diffusion Kalman filter
Assume Kalman filters connected into a **diffusion network**, in which the network nodes communicate only locally, i.e., only with their adjacent neighbors. Employing the adapt-then-combine (ATC) diffusion strategy, the filtering proceeds as follows:
- **local prediction** - the traditional Kalman filter prediction step,
- **adaptation** - incorporates its own and its neighbors' observations (e.g., sensor readings),
- **combination** - create a new estimate by combining the neighbors' estimates (the estimates of <!-- $x_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_t"> can be, for example, probability distributions).

## Partially heterogeneous models
The state-of-the-art methods for distributed Kalman filtering mostly assume spatial homogeneity of the state-space models. This work focuses on partially heterogeneous models, where the state-space of the less complex (or underspecified) models is a subspace of more complex model state-space. The simulation examples are run using the
1. Random-walk model (RWM)
2. Constant velocity model (CVM)
3. Constant acceleration model (CAM)

The models ordered by complexity are:
<!-- $$
\text{RWM} \leq \text{CVM} \leq \text{CAM}.
$$ -->

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctext%7BRWM%7D%20%5Cleq%20%5Ctext%7BCVM%7D%20%5Cleq%20%5Ctext%7BCAM%7D."></div>

Naturally, in an isolated environment with no collaboration among the nodes, the estimation performance and stability of the more complex models is significantly better than that of the underspecified models. The goal is to improve the estimation performance and stability of the RWM model by collaboration among the nodes.

The thesis proposes a modification to the existing diffusion Kalman filter. The local prediction and adaptation steps from the diffusion Kalman filter are preserved. The combination step is slightly modified, only estimates from neighbors of the same or better complexity are combined. In addition, a simple node failure detection method is proposed to further improve the performance of the underspecified models. Each node checks for inconsistency between its own estimate and the estimates of its neighbors (here Euclidean distance is used), when a failure is detected, the node is reset and reinitialized with a new estimate (in this case - an arithmetic mean of the neighbors' estimates).

# Results