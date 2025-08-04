import torch
from torch.optim import Optimizer
import math

class RegularizedLBFGS(Optimizer):
    def __init__(self, params, lr=1.0, max_iter=20, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10, damping=1e-4):
        defaults = dict(lr=lr, 
                        max_iter=max_iter, 
                        tolerance_grad=tolerance_grad,
                        tolerance_change=tolerance_change, 
                        history_size=history_size,
                        damping=damping)
        super(RegularizedLBFGS, self).__init__(params, defaults)

    def step(self, closure):
        if closure is None:
            raise RuntimeError("RegularizedLBFGS requires a closure that reevaluates the model.")

        loss = closure()

        # Gather parameters and gradients
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)

        # Flatten parameters and gradients
        flat_params = torch.cat([p.data.view(-1) for p in params])
        flat_grads = torch.cat([g.data.view(-1) for g in grads])

        # Settings
        lr = group['lr']
        max_iter = group['max_iter']
        tol_grad = group['tolerance_grad']
        tol_change = group['tolerance_change']
        history_size = group['history_size']
        damping = group['damping']

        n = flat_params.numel()

        # Initialize inverse Hessian approximation as identity
        Hk = torch.eye(n, device=flat_params.device)

        old_params = flat_params.clone()
        old_grads = flat_grads.clone()

        # Main BFGS loop
        for _ in range(max_iter):
            # Compute search direction
            pk = - Hk @ flat_grads

            # Line search (simplified: fixed step size lr)
            flat_params_new = flat_params + lr * pk

            # Update model parameters
            offset = 0
            for p in params:
                numel = p.numel()
                p.data.copy_(flat_params_new[offset:offset + numel].view_as(p))
                offset += numel

            # Evaluate new loss and gradients
            loss_new = closure()
            new_grads = torch.cat([p.grad.view(-1) for p in params])

            # Check convergence on gradient
            if new_grads.norm() < tol_grad:
                break

            # Update s and y
            s = (flat_params_new - flat_params).detach()
            y = (new_grads - flat_grads).detach()

            # Regularize y to ensure positive curvature (if necessary)
            if torch.dot(s, y) < 1e-10:
                y = y + damping * s

            rho = 1.0 / torch.dot(y, s)
            I = torch.eye(n, device=flat_params.device)
            V = I - rho * torch.outer(s, y)
            Hk = V @ Hk @ V.T + rho * torch.outer(s, s)

            # Update parameters
            flat_params = flat_params_new
            flat_grads = new_grads

            # Check for parameter convergence
            if torch.norm(flat_params - old_params) < tol_change:
                break

            old_params = flat_params.clone()

        return loss
