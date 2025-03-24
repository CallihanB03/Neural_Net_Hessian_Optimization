# Optimization-Based Neural Network Training with Hessian Methods


## Motivation
The motivation for this project comes from approaching the training of neural networks an iterative algorithm in computational mathematics. In traditional neural network training, weights are updated during back propagation via gradient descent. That is, at iteration $i+1$, the parameters $\theta_{i+1}$ are updated according to

$$
\theta_{i+1} = \theta_{i} - \nabla \mathcal{L}(\theta_{i})
$$

where $\mathcal{L}(\theta_{i})$ is the loss function evaluated for the set of parameters at iteration $i$ and $\nabla$ is the gradient operator. The gradient descent algorithm has order $\mathcal{O}(\theta)$ convergence. 


In this project, I explore the use of Hessian methods to speed up the training of neural networks. By defining the gradient update as the following

$$
\theta_{i+1} = \theta_{i} - H^{-1}\nabla\mathcal{L}(\theta_{i})
$$

where $H^{-1}$ is the inverse of the Hessian matrix, we are able to define neural network training as a Newton optimization problem with quadratic convergence of $\mathcal{O}(\theta^{2})$, making it converge significantly quicker than traditional gradient descent methods. 


## Challenges
Unfortunately, this approach presents a couple of challenges.

#### 1. Computational Expense
The computation of the Hessian inverse matrix $H^{-1}\nabla\mathcal{L}(\theta)$ is $\mathcal{O}(n^{3})$ making its computation impractical for large models with hundreds of thousands to millions of parameters.


#### 2. Hessian Not Positive Definite
A positive definite Hessian at a stationary point on the loss surface indicates that the function has a local minimum at that point. This is because the Hessian's eigenvalues are all positive, meaning that the function's curvature is "concave up" around the stationary point. If the Hessian is not positive definite, then the loss surface may not have a minimum.

#### 3. Ill-Conditioned Hessians
The condition number of the Hessian matrix, $H$, is defined as 

$$
\kappa(H) = \frac{\lambda_{max}}{\lambda_{min}}
$$

where $\lambda_{max}$ and $\lambda_{min}$ are the largest and smallest eigenvalues of $H$. A high $\kappa(H)$ implies that:

* The loss surface has highly elongated and narrow contours (poor curvature)

* Gradient-based updates can oscillate or move too slowly in some directions

* Second order methods may become unstable

In general, a Hessian matrix with a high condition number will result in slow convergence and may even prevent convergence entirely.

## Methods
To address the computational expense of computing the Hessian inverse $\mathcal{O}(n^{3})$, we consider Quasi Newton optimization algorithms to approximate $H^{-1}$. More specifically, we consider the [Broyden-Fletcher-Goldfarb-Shanno (BFGS)](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm) and [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) algorithms. These Quasi Newton algorithms work by approximating the Hessian matrix at every iteration. To briefly describe BFGS, consider $\mathcal{L}: \mathbb{R}^{n} \rightarrow \mathbb{R}$ to be the model's loss function, where $n$ is the number of parameters in the model, and $\mathcal{L}(\theta_{i})$ to be the model's loss at iteration $i$. Let $\mathcal{G}$ to be the second order approximation of $\mathcal{L}$ such that

$$
\mathcal{G}(\theta_{i}) \approx \mathcal{L}(\theta_{i}) + (\theta - \theta_{i})^{\intercal} \nabla \mathcal{L}(\theta_{i}) + \frac{1}{2}(\theta - \theta_{i})^{2}B_{i}\mathcal{L}(\theta_{i}) + \mathcal{O}((\theta - \theta_{i})^{3})
$$

where $B_{i}$ is an approximation of the Hessian matrix at iteration $i$. Then at iteration $i+1$, we seek to find a $B_{i+1}$ matrix such that $B_{i+1}$ abides by the following constraints: 

#### Constraint: B^{-1} is symmetric
To accurately approximate the Hessian, we penalize $B_{i+1}^{-1}$ for non-symmetry. This is because the Hessian matrix is a symmetric matrix.

$$
\begin{bmatrix}
\frac{\partial^{2}\mathcal{L}}{\partial^{2} \theta_{1}^{(i)}} & \dots & \dots & \frac{\partial^{2}\mathcal{L}}{\partial \theta_{1}^{(i)} \partial \theta_{n}^{(i)}}\\
\vdots & \ddots &  & \vdots\\
\vdots &  & \ddots & \vdots\\
\frac{\partial^{2}\mathcal{L}}{\partial \theta_{n}^{(i)} \partial \theta_{1}^{(i)}} & \dots & \dots & \frac{\partial^{2}\mathcal{L}}{\partial^{2} \theta_{n}^{(i)}}
\end{bmatrix}
$$

Here I use supperscripts on $\theta$ to denote the iteration and subscriptions on $\theta$ to denote a specific paramter.

#### Constraint: $ \nabla \mathcal{G}(\theta_{i}) = \nabla \mathcal{L}(\theta_{i})$
This constraint follows from the definition of $\mathcal{G}$ being a second order approximation.

<style>
  .bottom-three {
     margin-bottom: 10cm;
  }
</style>

#### Constraint: minimize $\|W(B^{-1}_{i+1} - B_{i}^{-1})W\|_{F}$
This constraint may not seem as obvious. In each iteration, we want $B_{i+1}$ to be "close" to $B_{i}$. We can do this by minimizing the Frobenius Norm between $B$ at subsequent iterations. That is, we can find $B_{i+1}$ that minimizes $\| B_{i+1} - B_{i} \|_{F}$. However, this approach has an important limitation in that it does not guarantee $B_{i+1}$ is positive definite. To guarantee $B_{i+1}$ is positive definite, we consider the [Davidon-Fletcher-Powell (DFP)](https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula). The DFP algorithm chooses a weighting matrix $W$ that enforces $B_{i+1}$ to be positive definite.

### BFGS
As previously stated, the BFGS algorithm picks a matrix $B_{i+1}$ that satisfies the above conditions. It's worth noting that this method can actually be performed analytically, resulting in explicit update formulas that are relatively less expensive to compute with a time complexity of $\mathcal{O}(n^{2})$. Importantly, no iterative methods are required for BFGS since it consists of a few vector and matrix calculations. 

### L-BFGS
The limited-memory BFGS algorithm outperforms the traditional BFGS algorithm by 



### Results
Results Placeholder

### Discussion
Discussion Placeholder

### Conclusion
Conclusion Placeholder