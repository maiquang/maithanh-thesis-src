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
where <!-- $x_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_t"> is the latent state, <!-- $y_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y_t"> is an observable output, <!-- $u_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=u_t"> is a **known** input variable, and <!-- $A_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=A_t">, <!-- $B_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=B_t">, <!-- $H_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_t"> are known matrices of compatible dimensions. Noise variables <!-- $w_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=w_t"> and <!-- $v_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=v_t"> are zero-mean, mutually independent and identically distributed.

## Kalman filter

## Diffusion Kalman filter

## Partially heterogeneous models


# Results