import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from pytorch_lightning import LightningModule
from torchmdnet.models import create_model, load_model


class LNNP(LightningModule):
    def __init__(self, hparams, mean=None, std=None, atomref=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, hparams=self.hparams)
        else:
            self.model = create_model(self.hparams)

        if atomref is not None:
            self.model.output_network.set_atomref(atomref)
        if mean is not None:
            self.model.output_network.mean = mean
        if std is not None:
            self.model.output_network.std = std

        self.losses = None
        self._reset_losses_dict()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.hparams.lr_factor,
                                      patience=self.hparams.lr_patience,
                                      min_lr=self.hparams.lr_min)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'interval': 'epoch',
                        'frequency': 1} 
        return [optimizer], [lr_scheduler]

    def forward(self, z, pos, batch=None):
        return self.model(z, pos, batch=batch)

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, 'train')

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, mse_loss, 'val')
        # test step
        return self.step(batch, l1_loss, 'test')

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, 'test')

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == 'train' or self.hparams.derivative):
            pred = self(batch.z, batch.pos, batch.batch)

        loss_y, loss_dy = 0, 0
        if self.hparams.derivative:
            pred, deriv = pred

            if 'y' not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0

            # force/derivative loss
            loss_dy = loss_fn(deriv, batch.dy)

            if self.hparams.force_weight > 0:
                self.losses[stage + '_dy'].append(loss_dy.detach())

        if 'y' in batch:
            # energy/prediction loss
            loss_y = loss_fn(pred, batch.y)
            
            if self.hparams.energy_weight > 0:
                self.losses[stage + '_y'].append(loss_y.detach())

        # total loss
        loss = loss_y * self.hparams.energy_weight + loss_dy * self.hparams.force_weight

        self.losses[stage].append(loss.detach())
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_steps))

            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, 'test_dataset') and len(dm.test_dataset) > 0:
            should_reset = (self.current_epoch % self.hparams.test_interval == 0 or
                            (self.current_epoch - 1) % self.hparams.test_interval == 0)
            if should_reset:
                # reset validation dataloaders before and after testing epoch, which is faster
                # than skipping test validation steps by returning None
                self.trainer.reset_val_dataloader(self)

    def validation_epoch_end(self, validation_step_outputs):
        if self.global_step > 0:
            # construct dict of logged metrics
            result_dict = {
                'epoch': self.current_epoch,
                'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                'train_loss': torch.stack(self.losses['train']).mean(),
                'val_loss': torch.stack(self.losses['val']).mean(),
            }

            # add test loss if available
            if len(self.losses['test']) > 0:
                result_dict['test_loss'] = torch.stack(self.losses['test']).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses['train_y']) > 0 and len(self.losses['train_dy']) > 0:
                result_dict['train_loss_y'] = torch.stack(self.losses['train_y']).mean()
                result_dict['train_loss_dy'] = torch.stack(self.losses['train_dy']).mean()
                result_dict['val_loss_y'] = torch.stack(self.losses['val_y']).mean()
                result_dict['val_loss_dy'] = torch.stack(self.losses['val_dy']).mean()

                if len(self.losses['test']) > 0:
                    result_dict['test_loss_y'] = torch.stack(self.losses['test_y']).mean()
                    result_dict['test_loss_dy'] = torch.stack(self.losses['test_dy']).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {'train': [], 'val': [], 'test': [],
                       'train_y': [], 'val_y': [], 'test_y': [],
                       'train_dy': [], 'val_dy': [], 'test_dy': []}
