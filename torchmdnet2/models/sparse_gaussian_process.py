from torch_scatter import scatter
import torch
from torch import nn

from .utils import Model



class SGPRModel(Model):
    def __init__(self, representation, kernel, weights=None, derivative='forces'):
        super(SGPRModel, self).__init__()
        self.save_hyperparameters()

        self.calculator = representation
        self.kernel = kernel

        n_weights = self.kernel.n_sparse_point
        if weights is None:
            weights = torch.ones((n_weights,1), dtype=kernel.sparse_points.dtype)/n_weights
        else:
            self.weights = nn.Parameter(weights, requires_grad=True)
        self.property = 'energy'
        self.contributions = None
        self.create_graph = True
        self.derivative = derivative
        self.stress = None
    #     self.automatic_optimization = False
    # def configure_optimizers(self):
    #     line_search_fn='strong_wolfe'
    #     # line_search_fn=None
    #     optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-3, max_iter=100,
    #                                     tolerance_grad=1e-5,
    #                                     tolerance_change=1e-5,
    #                                     line_search_fn=line_search_fn )
    #     return optimizer

    # def training_step(self, batch, batch_idx):
    #     opt = self.optimizers()
    #     # loss = self.step(batch, 'training')
    #     # opt.zero_grad()
    #     # self.manual_backward(loss, retain_graph=True)
    #     # loss.backward(retain_graph=True)
    #     def closure():
    #         loss = self.step(batch, 'training')
    #         opt.zero_grad()
    #         self.manual_backward(loss, retain_graph=True)
    #         # loss.backward(retain_graph=True)
    #         return loss

    #     opt.step(closure=closure)

    def step(self, data, stage):
        with torch.set_grad_enabled(self.derivative is not None):
            pred = self(data)

        losses = []
        facs = {'forces': 1000., 'energy': 1.}
        for k in pred.keys():
            ll = facs[k] * (pred[k] - data[k]).pow(2).mean()
            losses.append(ll)

        loss = torch.stack(losses).sum()
        # Add sync_dist=True to sync logging across all GPU workers
        self.log(f'{stage}_loss', loss.sqrt(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss


    def forward(self, data):
        if self.derivative is not None:
            data.pos.requires_grad_(True)

        features = self.calculator(data)
        Kmat = self.kernel(features, data)

        y = Kmat @ self.weights
        if self.derivative is not None:
            dy_dr = -torch.autograd.grad(
                y,
                data.pos,
                grad_outputs=torch.ones_like(y),
                retain_graph=self.training,
            )[0]

            return {self.property:y, self.derivative:dy_dr}
        else:
            return {self.property:y}
