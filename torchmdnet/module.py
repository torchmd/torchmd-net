import os
from functools import partial
import pytorch_lightning as pl
from torch_geometric.data import DataLoader

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from torchmdnet import datasets
from torchmdnet.utils import make_splits, TestingContext
from torchmdnet.data import Subset, AtomrefDataset
from torchmdnet.models import create_model


class LNNP(pl.LightningModule):
    def __init__(self, hparams):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            raise NotImplementedError()

        self.model = create_model(self.hparams)
        self.losses = None

    def setup(self, stage):
        if self.hparams.dataset == 'custom':
            self.dataset = datasets.Custom(
                self.hparams.coord_files,
                self.hparams.embed_files,
                self.hparams.energy_files,
                self.hparams.force_files
            )
        else:
            self.dataset = getattr(datasets, self.hparams.dataset)(
                self.hparams.dataset_root,
                label=self.hparams.label
            )

        if hasattr(self.dataset, 'get_atomref'):
            self.dataset = AtomrefDataset(self.dataset)

        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            self.hparams.val_ratio,
            self.hparams.test_ratio,
            self.hparams.seed,
            os.path.join(self.hparams.log_dir, 'splits.npz'),
            self.hparams.splits,
        )
        print(f'train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}')

        self.has_y = 'y' in self.dataset[0]
        self.has_dy = 'dy' in self.dataset[0]

        assert self.hparams.derivative == self.has_dy, 'Dataset has to contain "dy" if "derivative" is true.'

        self.train_dataset = Subset(self.dataset, idx_train)
        self.val_dataset = Subset(self.dataset, idx_val)
        self.test_dataset = Subset(self.dataset, idx_test)

        self._reset_losses_dict()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min
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

    def step(self, batch, loss_fn, stage):
        batch = batch.to(self.device)

        with torch.set_grad_enabled(stage == 'train' or self.hparams.derivative):
            pred = self(batch.z, batch.pos, batch.batch)

        loss = 0
        if self.hparams.derivative:
            pred, deriv = pred

            if not self.has_y:
                # "use" both outputs of the model's forward function but discard the first to only use the derivative and
                # avoid 'RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.',
                # which otherwise get's thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0

            # force/derivative loss
            loss = loss + loss_fn(deriv, batch.dy) * self.hparams.force_weight

        if self.has_y:
            # energy/prediction loss
            loss = loss + loss_fn(pred, batch.y) * self.hparams.energy_weight

        self.losses[stage].append(loss.detach())

        # PyTorch Lightning requires this in order for ReduceLROnPlateau to work
        self.log(f'{stage}_loss', loss.detach().cpu())
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_steps))

            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, 'train')

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, 'val')

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, 'test')

    def validation_epoch_end(self, validation_step_outputs):
        if self.global_step > 0:
            result_dict = {'epoch': self.current_epoch, 'lr': self.trainer.optimizers[0].param_groups[0]['lr']}
            result_dict['train_loss'] = torch.tensor(self.losses['train']).mean()
            result_dict['val_loss'] = torch.tensor(self.losses['val']).mean()

            if self.current_epoch % self.hparams.test_interval == 0:
                with TestingContext(self):
                    self.trainer.run_evaluation()
                result_dict['test_loss'] = torch.tensor(self.losses['test']).mean()

            self.log_dict(result_dict)
        self._reset_losses_dict()

    def _get_dataloader(self, dataset, stage):
        if stage == 'train':
            batch_size = self.hparams.batch_size
            shuffle = True
        elif stage in ['val', 'test']:
            batch_size = self.hparams.inference_batch_size
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def _reset_losses_dict(self):
        self.losses = {'train': [], 'val': [], 'test': []}
