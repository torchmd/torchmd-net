from os.path import join
from tqdm import tqdm
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from torch_scatter import scatter

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from torchmdnet import datasets
from torchmdnet.utils import make_splits
from torchmdnet.data import Subset
from torchmdnet.models import create_model, load_model


class LNNP(pl.LightningModule):
    def __init__(self, hparams):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, hparams=self.hparams)
        else:
            self.model = create_model(self.hparams)

        self.losses = None

    def setup(self, stage):
        if self.hparams.dataset == 'Custom':
            self.dataset = datasets.Custom(
                self.hparams.coord_files,
                self.hparams.embed_files,
                self.hparams.energy_files,
                self.hparams.force_files
            )
        else:
            self.dataset = getattr(datasets, self.hparams.dataset)(
                self.hparams.dataset_root,
                dataset_arg=self.hparams.dataset_arg
            )

        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            self.hparams.val_ratio,
            self.hparams.test_ratio,
            self.hparams.seed,
            join(self.hparams.log_dir, 'splits.npz'),
            self.hparams.splits,
        )
        print(f'train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}')

        self.train_dataset = Subset(self.dataset, idx_train)
        self.val_dataset = Subset(self.dataset, idx_val)
        self.test_dataset = Subset(self.dataset, idx_test)

        if hasattr(self.dataset, 'get_atomref'):
            self.model.output_network.set_atomref(self.dataset.get_atomref())

        if self.hparams.standardize:
            self._standardize()

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
        dataloader_idx = args[0] if len(args) > 0 else 0
        if dataloader_idx == 0:
            return self.step(batch, mse_loss, 'val')
        elif dataloader_idx == 1:
            if self.current_epoch % self.hparams.test_interval == 0:
                # test only on certain epochs
                return self.step(batch, l1_loss, 'test')
        return None

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, 'test')

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == 'train' or self.hparams.derivative):
            pred = self(batch.z, batch.pos, batch.batch)

        loss = 0
        if self.hparams.derivative:
            pred, deriv = pred

            if 'y' not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0

            # force/derivative loss
            loss = loss + loss_fn(deriv, batch.dy) * self.hparams.force_weight

        if 'y' in batch:
            # energy/prediction loss
            loss = loss + loss_fn(pred, batch.y) * self.hparams.energy_weight

        # save loss for logging
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

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, 'training')

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, 'inference')]
        if len(self.test_dataset) > 0:
            loaders.append(self._get_dataloader(self.test_dataset, 'inference'))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, 'inference')

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

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _get_dataloader(self, dataset, stage):
        if stage == 'training':
            batch_size = self.hparams.batch_size
            shuffle = True
        elif stage == 'inference':
            batch_size = self.hparams.inference_batch_size
            shuffle = False
        else:
            raise ValueError(f'Unknown stage "{stage}". Please choose "training" or "inference".')

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def _standardize(self):
        if self.model.output_network.atomref is not None:
            pl.utilities.rank_zero_warn('Standardizing when using a dataset with '
                                        'atomrefs likely leads to unwanted behaviour.')

        data = tqdm(self._get_dataloader(self.train_dataset, 'inference'),
                    desc='computing mean and std')
        ys = torch.cat([batch.y.clone() for batch in data])

        self.model.output_network.mean = ys.mean()
        self.model.output_network.std = ys.std()

    def _reset_losses_dict(self):
        self.losses = {'train': [], 'val': [], 'test': []}
