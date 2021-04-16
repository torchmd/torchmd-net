import os
from functools import partial
import pytorch_lightning as pl

from torch_geometric.nn.models.schnet import qm9_target_dict
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from utils import make_splits, TestingContext
from data import Subset, AtomrefDataset, CGDataset
from torchmd_gn import TorchMD_GN


class LNNP(pl.LightningModule):
    def __init__(self, hparams):
        super(LNNP, self).__init__()
        self.hparams = hparams

        if self.hparams.load_model:
            raise NotImplementedError()
        else:
            self.model = TorchMD_GN(
                hidden_channels=self.hparams.embedding_dimension,
                num_filters=self.hparams.num_filters,
                num_interactions=self.hparams.num_interactions,
                num_rbf=self.hparams.num_rbf,
                rbf_type=self.hparams.rbf_type,
                trainable_rbf=self.hparams.trainable_rbf,
                activation=self.hparams.activation,
                neighbor_embedding=self.hparams.neighbor_embedding,
                cutoff_lower=self.hparams.cutoff_lower,
                cutoff_upper=self.hparams.cutoff_upper,
                derivative=self.hparams.derivative
            )

        self.losses = None

    def setup(self, stage):
        if self.hparams.data:
            label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
            label_idx = label2idx[self.hparams.label]
            self.dataset = QM9(self.hparams.data, transform=partial(LNNP._filter_label, label_idx=label_idx))
            self.dataset = AtomrefDataset(self.dataset, self.dataset.atomref(label_idx))
        elif self.hparams.coords and self.hparams.forces and self.hparams.embed:
            self.dataset = CGDataset(self.hparams.coords, self.hparams.forces, self.hparams.embed)
        else:
            raise ValueError('Please provide either a QM9 database path or paths to coordinates, forces and emebddings.')

        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            self.hparams.val_ratio,
            self.hparams.test_ratio,
            self.hparams.seed,
            os.path.join(self.hparams.log_dir, 'splits.npz'),
            self.hparams.splits,
        )
        print(f'train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}')

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
        if self.hparams.derivative:
            _, pred = pred

        loss = loss_fn(pred, batch.y)
        self.losses[stage].append(loss.detach())

        if stage == 'val':
            # PyTorch Lightning requires this in order for ReduceLROnPlateau to work
            self.log('val_loss', loss.detach().cpu())
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_steps))

            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)

        # zero_grad call might be unnecessary here if we have Trainer(..., enable_pl_optimizer=True)
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
                    self.trainer.run_test()
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

    def _filter_label(batch, label_idx):
        batch.y = batch.y[:,label_idx].unsqueeze(1)
        return batch
