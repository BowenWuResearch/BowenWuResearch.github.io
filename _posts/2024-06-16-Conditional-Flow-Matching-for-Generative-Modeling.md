This is my personal understanding about the paper [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

## Introduction
In this deep learning era, generative modeling has been researched for about 10 years. Therea are numerous models with neat math deriviations related to log-likelihood estimation, including generative adversarial networks, variational autoencoders, normalizing flows, autoregressive models and diffusion models.

Interestingly, although diffusion models are derived from log-likelihood initially, it has been found that diffusion models are strongly related to differential equations, for both stochastic and ordinal. Furthermore, a trained diffusion model can be used to estimate the gradient of the log-likelihood of the data it is trained on, which is similar to vector fields that generates a probability path from one distribution to another.

The idea of modeling vector field for generative modeling has originated a while ago, where a probability path between a prior distribution and the data distribution is learned using a neural network. The trained networks can be used to estimate the vector field that generates this probability path, which then will be used to move any point from the prior distribution to the data distribution as the generation process.

In order to train such a network, we must have analytical solution for the vector field we want to model. However, this is expensive and often infeasible, as the vector field requires integral on all samples in the data:

$$
u_t(x) = \int u_t(x|x_1) q(x_1) dx_1,
$$

where $x_1$ represents each sample in the data. Therefore, it is hard to train a model using this vector field as ground truth.

Considering that common deep learning training algorithms can use only one sample in the data to compute loss and update parameters, can this vector field $u_t(x)$ be conditioned on only one sample as $u_t(x \lvert x_1)$ such that a model trained to match $u_t(x \lvert x_1)$ is equavalent or at least approximate $u_t(x)$? If this can be achieved, we will be able to efficiently train a vector field as a generative model, and the answer is **YES**, by desinging an appropriate loss function, called Conditional Flow Matching (CFM).

## CFM Loss
Denoting $v_t(x)$ as the network to be trained, the Flow Matching (FM) loss can be defined as follows:

$$
L_{FM} = E_{t, p_t(x)}[||v_t(x) - u_t(x)||^2],
$$

where $p_t(x)$ is probability path and $u_t(x)$ is the vector field that generates the probability path.

The gradient provided by this loss function can be used to optimize a model to output $u_t(x)$, which generates the probability path $p_t(x)$ thus can be used as generative model. However, it is not trivial to define a probability path as well as its generating vector field as there are arbitrarily infinite choices and it requries integral on all data points.

Instead, a flow matching loss conditioned on each data point, i.e., CFM loss, is defined as follows:

$$
L_{CFM} = E_{t, q(x_1), p_t(x|x_1)}[||v_t(x) - u_t(x|x_1)||^2],
$$

where $p_t(x \lvert x_1)$ is the contidional probability path and $u_t(x \lvert x_1)$ is the conditional vector field.

Before using it, it is unclear if CFM loss can be used to train a model that also minimizes FM loss. This can be verified by comparing their gradients with respect to the model.

$$
\begin{aligned}
L_{FM}  & = E_{t, p_t(x)}[||v_t(x)||^2 - 2<v_t(x), u_t(x)> + ||u_t(x)||^2] \\
L_{CFM} & = E_{t, q(x_1), p_t(x|x_1)}[||v_t(x)||^2 - 2<v_t(x), u_t(x|x_1)> + ||u_t(x|x_1)||^2]
\end{aligned}
$$

Since the gradient of $u_t$ with respect to the model is 0, we can only compare the first two terms.
For the first term, we can condition $x$ on $x_1$ to match them as follows:
$$
\begin{aligned}
E_{t, p_t(x)}[||v_t(x)||^2] & = E_{t}[||\int v_t(x) p_t(x) dx||^2] \\
& = E_{t}[||\iint v_t(x) p_t(x|x_1) q(x_1) dx_1 dx||^2] \\
& = E_{t, q(x_1)}[||\int v_t(x) p_t(x|x_1) dx||^2] \\
& = E_{t, q(x_1), p_t(x|x_1)}[||v_t(x)||^2].
\end{aligned}
$$

For the second term, we need to define the relationship between $u_t(x)$ and $u_t(x \lvert x_1)$ to match them. If we only consider to match $L_{CFM}$ with $L_{FM}$, there will be arbitrary many choices. However, remember that $u_t(x)$ needs to generate the probability path $p_t(x)$, which has to be satisfied by the parameterization of $u_t(x)$ using $u_t(x \lvert x_1)$.

A probability path $p_t(x)$ can be conditioned on data points $x_1$ as follows: 

$$
p_t(x) = \int p_t(x|x_1) q(x_1) dx_1,
$$ 

where $p_t(x \lvert x_1)$ is the conditional probability path. Now, we want to parameterize a vector field $u_t(x)$ using its conditional vector field $u_t(x \lvert x1)$, such that $u_t(x)$ generates $p_t(x)$. If this can be achieved, we will be able to use $u_t(x \lvert x1)$ to generate $p_t(x)$.

In order to find the parameterization, it is necessary to know how can a vector field be justified that it generates a probability path. Fortunatey, the continuity equation can be used. It states that: a vector field $u_t(x)$ generates a probability path $p_t(x)$ when the following holds: 

$$
\frac{dp_t(x)}{dt} + div(p_t(x) u_t(x)) = 0,
$$ 

where $div=\sum_i^d\frac{\partial}{\partial x^i}$ is defined with respect to a spatial variable $x=(x^0, ..., x^d)$. Note that for a specific data point $x_1$, its vector field $u_t(x \lvert x_1)$ generates its probability path $p_t(x \lvert x_1)$, thus the following holds:

$$
\frac{dp_t(x|x_1)}{dt} + div(p_t(x|x_1) u_t(x|x_1)) = 0.
$$

The continuity equation can be utilized to derived the parameterization for $u_t(x)$ conditioned on $u_t(x \lvert x_1)$ such that it generates the probability path $p_t(x)$. The following derives the parameterization of $u_t(x)$ starting from the continuity equation:

$$
\frac{dp_t(x)}{dt} = \frac{d (\int p_t(x|x_1) q(x_1) dx_1)}{dt} = \int \frac{dp_t(x|x_1)}{dt}  q(x_1) dx_1 \\
= \int - div(p_t(x|x_1) u_t(x|x_1)) q(x_1) dx_1 = - div (\int p_t(x|x_1) u_t(x|x_1) q(x_1) dx_1).
$$

For the continuity eqaution to holds, we need $p_t(x) u_t(x) = \int p_t(x \lvert x_1) u_t(x \lvert x_1) q(x_1) dx_1$. To achieve this, we can parameterize $u_t(x)$ as follows:

$$
u_t(x) := \frac{\int p_t(x|x_1) u_t(x|x_1) q(x_1) dx_1}{p_t(x)} = \int u_t(x|x_1) \frac{ p_t(x|x_1) q(x_1)}{p_t(x)} dx_1.
$$

As a result, this vector field $u_t(x)$ generates the probability path $p_t(x)$ when conditioning on each sample $x_1$ in the data.

With this paramterization, the second term of $L_{FM}$ becomes:

$$
E_{t, p_t(x)}[<v_t(x), \int u_t(x|x_1) \frac{ p_t(x|x_1) q(x_1)}{p_t(x)} dx_1>] = E_{t}[<v_t(x), \iint u_t(x|x_1) \frac{ p_t(x|x_1) q(x_1)}{p_t(x)} dx_1 p_t(x) dx>] \\
= E_{t}[<v_t(x), \iint u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1 dx>] = E_{t, q(x_1), p_t(x|x_1)}[<v_t(x), u_t(x|x_1)>], 
$$

which matches the second term of $L_{CFM}$.

The above indicates that $\nabla_v L_{FM} = \nabla_v L_{CFM}$. Training a model using $L_{CFM}$ is equavalent to using $L_{FM}$.

## Gaussian Probability Path
When conditoned on $x_1$, the probability path is defined as:

$$
p_t(x \mid x_1) = N(x; \mu_t(x_1), \sigma_t(x_1)).
$$

A simple affine transformation using $\mu_t$ and $\sigma_t$ can be as follows to define the flow:

$$
\psi_t(x \mid x_1) = \mu_t(x_1) + \sigma_t(x_1) x,
$$

where $x \sim N(0, 1)$.

Reparameterization using $p(x_0)$ to replace $p(x \mid x_t)$, the LCM loss function becomes:

$$
L_{CFM} = E_{t, q(x_1), p_t(x_0)}[||v_t(\psi_t(x_0)) - u_t(\psi_t(x_0 \mid x_1))||^2],
$$

where $u_t(\psi_t(x_0 \mid x_1)) = \frac{d\psi_t(x_0 \mid x_1)}{dt}$.

## Linear Interpolation as Optimal Transport
We can define $\mu_t$ and $\sigma_t$ as follows:

$$
\mu_t(x_1) = t x_1; \sigma_t(x_1) = 1 - (1 - \sigma_{min})t,
$$

which leads to linear interpolation between $x_0$ and $x_1$.

According to the paper, this results in a flow that realize the optimal transport from $p(x_0)$ to $p(x_1)$.

The flow becomes:

$$
\psi_t(x_0 \mid x_1) = t x_1 + (1 - (1 - \sigma_{min})t) x_0
$$

Its derivative:

$$
\frac{d\psi_t}{dt} = x_1 - (1 - \sigma_{min}) x_0
$$

Loss function becomes:

$$
L_{CFM} = E_{t, q(x_1), p_t(x_0)}[||v_t(\psi_t(x_0)) - (x_1 - (1 - \sigma_{min}) x_0)||^2],
$$

## Sampling
Suppose a flow $\psi_t$ corresponding to a probability path $p_t$ from $p(x0)$ to $q(x1$), we can use the derivative of the flow to iterative find the solution, given an initial value.

The derivative of the flow is defined as follows:

$$
\frac{d\psi_t}{dt} = lim_{\delta \rightarrow 0} \frac{x_{t+\delta} - x_t}{\delta}.
$$

The deduction is:

$$
x_{t+\delta} \leftarrow x_t + \delta \frac{d\psi_t}{dt}.
$$

In CFM, a model $v_t(x_t)$ is trained to match $u_t(x_t)$, i.e., $\frac{d\psi_t}{dt}$ at $t$. Therefore, the deduction using model becomes:

$$
x_{t+\delta} \leftarrow x_t + \delta v_t(x_t).
$$

In practice, we divide 1 by $N$ generation steps and use it as $\delta$ to approximate the derivative for the generation.


<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>