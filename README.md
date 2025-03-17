# Optimization-Based Neural Network Training with Hessian Methods


### Motivation
The motivation for this project comes from approaching neural network training as an iterative algorithm in computational mathematics. In traditional neural network training, weights are updated during back propagation via gradient descent. That is, at iteration $i+1$, the parameters $\theta_{i+1}$ are updated using

$$
\theta_{i+1} = \theta_{i} - \nabla \mathcal{L}(\theta_{i})
$$

where $\mathcal{L}(\theta_{i})$ is the loss function evaluated for the set of parameters at iteration $i$ and $\nabla$ is the gradient operator. The gradient descent algorithm has order $\mathcal{O}(\theta)$ convergence. 


In this project, I explore the use of Hessian methods to speed up the training of neural networks. Specifically, I seek to define the gradient update as the following
$$
\theta_{i+1} = \theta_{i} - H^{-1}\nabla\mathcal{L}(\theta_{i})
$$

where $H$ is the Hessian matrix. This approach defines neural network training as a Newton optimization problem with quadratic convergence of $\mathcal{O}(\theta^{2})$, making it converge significantly quicker than traditional gradient descent methods.


### Drawbacks
There are two main drawbacks in using the Hessian inverse matrix for updating gradients.

##### 1) Computational Expense
The computation of the Hessian inverse matrix $H^{-1}\nabla\mathcal{L}(\theta)$ is $\mathcal{O}(n^{3})$ making its computation impractical for large datasets. 