import os
import pytorch_lightning as pl

from torch_geometric.nn.models.schnet import SchNet, qm9_target_dict
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from utils import make_splits


class LNNP(pl.LightningModule):
    def __init__(self, hparams):
        super(LNNP, self).__init__()
        self.hparams = hparams

        self.dataset = QM9('data/')

        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.hparams.label]

        if self.hparams.load_model:
            raise NotImplementedError()
        else:
            self.model = SchNet(
                hidden_channels=self.hparams.embedding_dimension,
                num_filters=self.hparams.num_filters,
                num_interactions=self.hparams.num_interactions,
                num_gaussians=self.hparams.num_rbf,
                cutoff=self.hparams.upper_cutoff,
                atomref=self.dataset.atomref(target=self.label_idx)
            )

        self.losses = None

    def setup(self, stage):
        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            self.hparams.val_ratio,
            self.hparams.test_ratio,
            self.hparams.seed,
            os.path.join(self.hparams.log_dir, 'splits.npz'),
            self.hparams.splits,
        )
        print(f'train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}')

        self.train_dataset = self.dataset[:20]#idx_train]
        self.val_dataset = self.dataset[:20]#idx_val]
        self.test_dataset = self.dataset[:20]#idx_test]

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
        pred = self(batch.z, batch.pos, batch.batch)
        loss = mse_loss(pred[:,0], batch.y[:,self.label_idx])
        self.losses['train'].append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch.z, batch.pos, batch.batch)
        loss = mse_loss(pred[:,0], batch.y[:,self.label_idx])
        self.losses['val'].append(loss.detach())
        return loss

    def test_step(self, batch, batch_idx):
        pred = self(batch.z, batch.pos, batch.batch)
        loss = l1_loss(pred[:,0], batch.y[:,self.label_idx])
        self.losses['test'].append(loss.detach())
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
        return self._get_dataloader(self.val_dataset, 'validation')

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, 'test')

    def validation_epoch_end(self, validation_step_outputs):
        if self.global_step > 0:
            result_dict = {'epoch': self.current_epoch, 'lr': self.trainer.optimizers[0].param_groups[0]['lr']}
            result_dict['train_loss'] = torch.tensor(self.losses['train']).mean()
            result_dict['val_loss'] = torch.tensor(self.losses['val']).mean()

            if self.current_epoch % self.hparams.test_interval == 0:
                # if the testing flag is not set in the trainer run_test uses validation_step instead of test_step
                self.trainer.testing = True
                result = self.trainer.run_test()
                self.trainer.testing = False
                result_dict['test_loss'] = torch.tensor(self.losses['test']).mean()

            self.log_dict(result_dict)
        self._reset_losses_dict()
        return {'val_loss': torch.stack(validation_step_outputs).mean()}

    def _get_dataloader(self, dataset, stage):
        if stage == 'train':
            batch_size = self.hparams.batch_size
            shuffle = True
        elif stage in ['validation', 'test']:
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
