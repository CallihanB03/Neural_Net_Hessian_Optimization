# Optimization-Based Neural Network Training with Hessian Methods


## Motivation
The motivation for this project comes from approaching the training of neural networks as an iterative algorithm in computational mathematics. In traditional neural network training, weights are updated during back propagation via gradient descent. That is, at iteration $i+1$, the parameters $\theta_{i+1}$ are updated according to

$$
\theta_{i+1} = \theta_{i} - \nabla \mathcal{L}(\theta_{i})
$$

where $\mathcal{L}(\theta_{i})$ is the loss function evaluated for the set of parameters at iteration $i$ and $\nabla$ is the gradient operator. The gradient descent algorithm has a linear order $\mathcal{O}(\theta)$ convergence. 


In this project, I explore the use of Hessian methods to speed up the training of neural networks. By defining the gradient update as

$$
\theta_{i+1} = \theta_{i} - H^{-1}\nabla\mathcal{L}(\theta_{i})
$$

where $H^{-1}$ is the inverse of the Hessian matrix, we are able to consider neural network training as a Newton optimization problem with quadratic convergence of $\mathcal{O}(\theta^{2})$, making it converge significantly quicker than traditional gradient descent methods. 


## Challenges
Unfortunately, this approach presents a couple of challenges.

#### 1. Computational Expense
The computation of the Hessian inverse matrix $H^{-1}\nabla\mathcal{L}(\theta)$ has $\mathcal{O}(\theta^{3})$ time complexity and $\mathcal{O}(\theta^{2})$ space complexity and making its computation impractical for large models with millions of parameters.


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

## Method
To address the computational expense of computing the Hessian inverse, we consider Quasi Newton optimization algorithms for approximation. More specifically, we consider the [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) algorithms. This Quasi Newton algorithm work by approximating the matrix-vector product $H^{-1} \nabla f(\theta)$. To describe the L-BFGS algorithm, let us first begin by describing the BFGS algorithm.

### BFGS
Consider a neural network with parameters $\theta \in \mathbb{R}^{n}$ and a loss function $f: \mathbb{R}^{n} \to \mathbb{R}$. Further, consider the following unconstrained optimization problem:
$$
\min_{\theta \in \mathbb{R}^{n}} f(\theta)
$$

From an initial guess $\theta_{0} \in \mathbb{R}^{n}$, an initial approximate inverted Hessian $H_{0}$, and a tolerance $\varepsilon$, the following is repeated as $\theta_{k} \to \arg \min_{\theta} f(\theta)$

$\text{For } k=0, \dots, max\_iter$
1. Obtain a direction $p_{k}$ by solving $$p_{k} = - H_{k} \nabla f(\theta_{k})$$
2. Perform a one-dimensional optimization (line search) to find an acceptable step size $\alpha_{k} \approx \arg \min f(\theta_{k} + \alpha_{k} p_{k})$
	* Here I choose to perform the Armijo line search:
		***Armijo Line Search Algorithm**
		1. Initialize $c, \beta, max\_ls$
		2. $loss\_initial \gets f(\theta_{k}^{i=0})$
		3. For $i = 1, \dots, max\_ls$
			1. $\theta_{k}^{i+1} \gets  \theta_{k}^{i} + \alpha_{k} p_{k}$
			2. $\text{if } f(\theta_{k}^{i+1}) < loss\_initial  + c \alpha_{k} \ \langle f(\theta_{k}^{i=0}), p_{k}\rangle$
				 $i^{(*)} \gets i$
				 break
			 3. $\alpha_{k} \gets \alpha_{k} + \beta$
	
3. Set $ $\theta_{k+1} \gets \theta_{k}^{i^{(*)}}$ and $s_{k} \gets \theta_{k+1} - \theta_{k}$ 
4. Set $y_{k} \gets \nabla f(\theta_{k+1}) - \nabla f(\theta_{k})$
5. Update the approximate inverse Hessian
	$\rho_{k} \gets \frac{1}{y_{k}^{\intercal} s_{k}}$
	$V_{k} \gets I - \rho_{k} \ s_{k} \otimes_{\text{outer}} y_{k}$
	$H_{k+1} \gets V_{k} H_{k} V_{k}^{\intercal} + \rho_{k} \ s_{k} \otimes_{\text{outer}} s_{k}$
6. $\text{if} \| \theta_{k+1} - \theta_{k} \| < \varepsilon$ 
		return $H_{k}$

The BFGS algorithm only requires a time complexity of $\mathcal{O}(\theta^{2})$, compared to the time complexity of direct computation $\mathcal{O}(\theta^{3})$, but it still requires an approximate inverse Hessian matrix and, thus, still has a prohibitively expensive space complexity of $\mathcal{O}(\theta^{2})$. 


### L-BFGS
Unlike the BFGS algorithm, the L-BFGS algorithm is matrix-free and does not require the direct computation of the approximate inverse Hessian matrix. Instead, the algorithm uses a memory mechanism to approximate the matrix vector product $H^{-1} \nabla f(\theta)$. Below is a description of the algorithm at interval $k$. 


$q \gets g_{k} = \nabla \mathcal{L}(w_{k})$

$\textbf{for } i=k-1, \dots, k-m \textbf{ do}$

$\hspace{1 cm} \alpha_{i} = \frac{s_{i}^{\intercal} q}{y_{i}^{\intercal} s_{i}}$

$\hspace{1 cm} q \gets q - \alpha_{i} y_{i}$

$\textbf{end for}$

$r \gets H_{0} q$

$\textbf{for } i = k-1, \dots, k - m \textbf{ do}$

$\hspace{1 cm} \beta = \frac{y_{i}^{\intercal} r}{y_{i}^{\intercal} s_{i}}$

$\hspace{1 cm} r \gets r + s_{i} (\alpha_{i} - \beta)$

$\textbf{end for}$

$\textbf{return } -r = - H_{k}g_{k}$

Although the L-BFGS algorithm still has a time complexity  of $\mathcal{O}(\theta^{2})$, because of the memory mechanism - $m$, the space complexity is $\mathcal{O}(\theta)$

## Results
I trained and evaulated a Convolutional Neural Network on the MNIST Fashion Dataset using both the Adaptive Moments (Adam) optimizer and the L-BFGS based optimizer. For both models, training continues until
1. Convergence (i.e. |$\mathcal{L}(\theta_{i+1}) - \mathcal{L}(\theta_{i}))| < \varepsilon$
2. $i = max\_iter$

### Adam Optimizer
Although the model heavily overfits the training data, the validation accuracies are still quite high and the model finishes with a respectable test accuracy of 90%.

![Adam Optimizer Performance](figures/CNN%20Validation%20Accuracy%20with%20Adam.png)

CNN Validation Accuracies using Adam Optimizer



### L-BFGS Based Optimizer
The model trained using the L-BFGS based optimizer also suffers from overfitting. This model was also very unstable.

There were many times where the model converged quicker using the L-BFGS optimizer than the Adam optimizer, however it is worth noting that in those instances:
1. The model's runtime using the L-BFGS based optimizer was significantly (~8.5 times) longer than when it used the Adam optimizer. This is likely due to the fact that Quasi-Newton update parameters still require a time complexity of $\mathcal{O}(\theta^{2})$ to compute. 
2. The CNN seems to converge to a local minimum using the L-BFGS based optimizer, as it performs worse than its Adam counterpart.

Overall, the CNN trained using the L-BFGS based optimizer achievees a test accuracy of about 70%. 

Another negative of the L-BFGS based optimization training is its instability. Likely due to the non-convexity of the loss surface and the high condition number of the Hessian, there were many different seeds that I trained the model on that resulted in mode collapse. 


![L-BFGS Optimizer Val Acc](figures/Collapsed%20Training.png)

CNN experiences collapse while training using L-BFGS based optimizer.

### Conclusion
In practice, it seems that Quasi-Newton methods for neural network training are not practical unless the loss surface maintains convexity which is not likely due to the strong non-convex nature of loss surfaces in machine learning. Even if the loss surface is (locally) convex, the additional runtime associated with the L-BFGS algorithm affirms to the conclusion that Quasi-Newton Methods are not practical for training large models. 
