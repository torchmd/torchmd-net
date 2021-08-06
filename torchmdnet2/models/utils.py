import os
import pytorch_lightning as pl

from torch_geometric.data import DataLoader

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from ..utils import make_splits, TestingContext


class Model(pl.LightningModule):
    def __init__(self, #model: pl.LightningModule,
    lr:float =1e-4, weight_decay:float=0, lr_factor:float=0.8,
    lr_patience:int=10, lr_min:float=1e-7, target_name='forces',
    lr_warmup_steps:int=0,
    test_interval:int=1,
):
        super(Model, self).__init__()
        #self.model = model
        self.losses = None
        # self.derivative = model.derivative
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_min = lr_min
        self.target_name = target_name
        self.lr_warmup_steps = lr_warmup_steps
        self.test_interval = test_interval

    def configure_optimizers(self):
        optimizer = AdamW(#self.model.parameters(),
        self.parameters(),
        lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min
        )
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'validation_loss',
                        'interval': 'epoch',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def training_step(self, data, batch_idx):
        loss = self.step(data, 'training')
        return loss


    def validation_step(self, data, batch_idx):
        loss = self.step(data, 'validation')
        return loss

    def test_step(self, data, batch_idx):
        loss = self.step(data, 'test')
        return loss


    def step(self, data, stage):
        with torch.set_grad_enabled(stage == 'train' or self.derivative):
            pred = self(data.z, data.pos, data.batch)

        loss = 0
        facs = {'forces': 1.}
        for k,fac in facs.items():
            loss += fac * (pred[k] - data[k]).pow(2).mean()

        # Add sync_dist=True to sync logging across all GPU workers
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

