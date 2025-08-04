from torch.optim import Optimizer, LBFGS

    

class LBFGS_based_opimizer(LBFGS):
    def __init__(self, params, lr=1.0, max_iter=20, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10, damping=1e-4):
        defaults = dict(lr=lr, 
                        max_iter=max_iter, 
                        tolerance_grad=tolerance_grad,
                        tolerance_change=tolerance_change, 
                        history_size=history_size,
                        damping=damping)
        super(LBFGS_based_opimizer, self).__init__(params, **defaults)
        self.batch_loss = None

    def step(self, closure):
        super(self.step(closure))

