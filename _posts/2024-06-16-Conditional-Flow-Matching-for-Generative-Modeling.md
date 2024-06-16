This is my personal understanding about the paper [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

## Introduction
In this deep learning era, generative modeling has been researched for about 10 years. Therea are numerous models with neat math deriviations related to log-likelihood estimation, including generative adversarial networks, variational autoencoders, normalizing flows, autoregressive models and diffusion models.

Interestingly, although diffusion models are derived from log-likelihood initially, it has been found that diffusion models are strongly related to differential equations, for both stochastic and ordinal. Furthermore, a trained diffusion model can be used to estimate the gradient of the log-likelihood of the data it is trained on, which is similar to vector fields that generates a probability path from one distribution to another.

The idea of modeling vector field for generative modeling has originated a while ago, where a probability path between a prior distribution and the data distribution is learned using a neural network. The trained networks can be used to estimate the vector field that generates this probability path, which then will be used to move any point from the prior distribution to the data distribution as the generation process.

In order to train such a network, we must have analytical solution for the vector field we want to model. However, this is expensive and often infeasible, as the vector field requires integral on the data: 
\begin{align}
u_t(x) = \int u_t(x|x_1) q(x_1) dx_1
\end{aligh}

where $x_1$ represents each data point. Therefore, it is hard to train a model using this vector field as ground truth.

Considering that common deep learning training algorithms can use only one sample in the data to compute loss and update parameters, can this vector field $u_t(x)$ be conditioned on only one sample as $u_t(x|x_1)$ such that a model trained to match $u_t(x|x_1)$ is equavalent or at least approximate $u_t(x)$? If this can be achieved, we will be able to efficiently train a vector field as a generative model, and the answer is **YES**, by desinging an appropriate loss function, called Conditional Flow Matching (CFM).

## CFM Loss
Denoting $v_t(x)$ as the network to be trained, the Flow Matching (FM) loss can be defined as follows:
$$\mathcal{L}_{FM} = E_{t, p_t(x)}[||v_t(x) - u_t(x)||^2],$$

where $p_t(x)$ is probability path and $u_t(x)$ is the vector field that generates the probability path.

The gradient provided by this loss function can be used to optimize a model to output $u_t(x)$, which generates the probability path $p_t(x)$ thus can be used as generative model. However, it is not trivial to define a probability path as well as its generating vector field as there are arbitrarily infinite choices and it requries integral on all data points.

Instead, a flow matching loss conditioned on each data point, i.e., CFM loss, is defined as follows:
$$\mathcal{L}_{CFM} = E_{t, q(x_1), p_t(x|x_1)}[||v_t(x) - u_t(x|x_1)||^2],$$

where $p_t(x|x_1)$ is the contidional probability path and $u_t(x|x_1)$ is the conditional vector field.

Before using it, it is unclear if $\mathcal{L}_{CFM}$ can be used to train a model that also minimizes $\mathcal{L}_{FM}$. This can be verified by comparing their gradients with respect to the model.
$$L_{FM}  = E_{t, p_t(x)}[||v_t(x)||^2 - 2 <v_t(x), u_t(x)> + ||u_t(x)||^2]$$

$$L_{CFM}  = E_{t, q(x_1), p_t(x|x_1)}[||v_t(x)||^2 - 2 <v_t(x), u_t(x|x_1)> + ||u_t(x|x_1)||^2]$$

Since the gradient of $u_t$ with respect to the model is 0, we can only compare the first two terms.
For the first term, we can condition $x$ on $x_1$ to match them as follows:
$$E_{t, p_t(x)}[||v_t(x)||^2] = E_{t}[||\int v_t(x) p_t(x) dx||^2]$$

$$ = E_{t}[||\iint v_t(x) p_t(x|x_1) q(x_1) dx_1 dx||^2] = E_{t, q(x_1), p_t(x|x_1)}[||v_t(x)||^2].$$

For the second term, we need to define the relationship between $u_t(x)$ and $u_t(x|x_1)$ to match them. If we only consider to match $L_{CFM}$ with $L_{FM}$, there will be arbitrary many choices. However, remember that $u_t(x)$ needs to generate the probability path $p_t(x)$, which has to be satisfied by the parameterization of $u_t(x)$ using $u_t(x|x_1)$.

The deriviation will take some time. For the contuinity of presentation, the deriviation is detailed in the next section. The appropriate parameterization is directly provided here as follows:
$$u_t(x) :=\int u_t(x|x_1) \frac{ p_t(x|x_1) q(x_1)}{p_t(x)} dx_1.$$

Then, the second term of $L_{FM}$ becomes:
$$E_{t, p_t(x)}[<v_t(x), \int u_t(x|x_1) \frac{ p_t(x|x_1) q(x_1)}{p_t(x)} dx_1>]$$

$$= E_{t}[<v_t(x), \iint u_t(x|x_1) \frac{ p_t(x|x_1) q(x_1)}{p_t(x)} dx_1 p_t(x) dx>]$$

$$= E_{t}[<v_t(x), \iint u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1 dx>]$$

$$= E_{t, q(x_1), p(x|x_1)}[<v_t(x), u_t(x|x_1)>], $$

which matches the second term of $L_{CFM}$.

The above indicates that $\nabla_v L_{FM} = \nabla_v L_{CFM}$. Training a model using $L_{CFM}$ is equavalent to using $L_{FM}$.

## Conditional Probability Path and Vector Field
A probability path $p_t(x)$ can be conditioned on data points $x_1$ as follows: 
$$p_t(x) = \int p_t(x|x_1) q(x_1) dx_1,$$ 

where $p_t(x|x_1)$ is the conditional probability path. Now, we want to parameterize a vector field $u_t(x)$ using its conditional vector field $u_t(x|x1)$, such that $u_t(x)$ generates $p_t(x)$. If this can be achieved, we will be able to use $u_t(x|x1)$ to generate $p_t(x)$.

In order to find the parameterization, it is necessary to know how can a vector field be justified that it generates a probability path. Fortunatey, the continuity equation can be used. It states that: a vector field $u_t(x)$ generates a probability path $p_t(x)$ when the following holds: 
$$\frac{dp_t(x)}{dt} + div(p_t(x) u_t(x)) = 0,$$ 

where $div=\sum_i^d\frac{\partial}{\partial x^i}$ is defined with respect to a spatial variable $x=(x^0, ..., x^d)$. Note that for a specific data point $x_1$, its vector field $u_t(x|x_1)$ generates its probability path $p_t(x|x_1)$, thus the following holds:
$$\frac{dp_t(x|x_1)}{dt} + div(p_t(x|x_1) u_t(x|x_1)) = 0.$$

The continuity equation can be utilized to derived the parameterization for $u_t(x)$ conditioned on $u_t(x|x_1)$ such that it generates the probability path $p_t(x)$. The following derives the parameterization of $u_t(x)$ starting from the continuity equation:
$$\frac{dp_t(x)}{dt} = \frac{d (\int p_t(x|x_1) q(x_1) dx_1)}{dt} = \int \frac{dp_t(x|x_1)}{dt}  q(x_1) dx_1$$

$$= \int - div(p_t(x|x_1) u_t(x|x_1)) q(x_1) dx_1 = - div (\int p_t(x|x_1) u_t(x|x_1) q(x_1) dx_1).$$

For the continuity eqaution to holds, we need $p_t(x) u_t(x) = \int p_t(x|x_1) u_t(x|x_1) q(x_1) dx_1$. To achieve this, we can parameterize $u_t(x)$ as follows:
$$u_t(x) := \frac{\int p_t(x|x_1) u_t(x|x_1) q(x_1) dx_1}{p_t(x)} = \int u_t(x|x_1) \frac{ p_t(x|x_1) q(x_1)}{p_t(x)} dx_1.$$

As a result, this vector field $u_t(x)$ generates the probability path $p_t(x)$.


<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>