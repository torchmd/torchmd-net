import os
import pytorch_lightning as pl

from torch_geometric.data import DataLoader

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from ..utils import make_splits, TestingContext


class Model(pl.LightningModule):
    def __init__(self, model: Model, lr=1e-4, weight_decay=0, lr_factor=0.8,
    lr_patience=10, lr_min=1e-7, target_name='forces',
    lr_warmup_steps=0,
    test_interval=1,
):
        super(Model, self).__init__()
        self.model = model
        self.losses = None
        self.derivative = model.derivative
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_min = lr_min
        self.target_name = target_name
        self.lr_warmup_steps = lr_warmup_steps
        self.test_interval = test_interval

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min
        )
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'interval': 'epoch',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def forward(self, z, pos, batch=None):
        return self.model(z, pos, batch=batch)

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, 'test')

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_loss', avg_loss.detach().cpu())

    def setup(self, stage):

        self._reset_losses_dict()

    def step(self, batch, loss_fn, stage):
        batch = batch.to(self.device)

        with torch.set_grad_enabled(stage == 'train' or derivative):
            pred = self(batch.z, batch.pos, batch.batch)
        if self.derivative:
            # "use" both outputs of the model's forward function but discard the first to only use the derivative and
            # avoid 'RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.',
            # which otherwise get's thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
            out, deriv = pred
            pred = deriv + out.sum() * 0

        loss = loss_fn(pred, batch[self.target_name])
        self.losses[stage].append(loss.detach())

        if stage == 'val':
            # PyTorch Lightning requires this in order for ReduceLROnPlateau to work
            self.log('val_loss', loss.detach().cpu())
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else args[2]
        if self.trainer.global_step < self.lr_warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.lr_warmup_steps))

            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        super().optimizer_step(*args, **kwargs)

        # zero_grad call might be unnecessary here if we have Trainer(..., enable_pl_optimizer=True)
        optimizer.zero_grad()

    def validation_epoch_end(self, validation_step_outputs):
        if self.global_step > 0:
            result_dict = {'epoch': self.current_epoch, 'lr': self.trainer.optimizers[0].param_groups[0]['lr']}
            result_dict['train_loss'] = torch.tensor(self.losses['train']).mean()
            result_dict['val_loss'] = torch.tensor(self.losses['val']).mean()

            if self.current_epoch % self.test_interval == 0:
                with TestingContext(self):
                    self.trainer.run_test()
                result_dict['test_loss'] = torch.tensor(self.losses['test']).mean()

            self.log_dict(result_dict)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {'train': [], 'val': [], 'test': []}


