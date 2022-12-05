import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from pytorch_lightning import LightningModule
from torchmdnet.models.model import create_model, load_model


class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()

        if "charge" not in hparams:
            hparams["charge"] = False
        if "spin" not in hparams:
            hparams["spin"] = False

        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": getattr(self.hparams, "lr_metric", "val_loss"),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(self, z, pos, batch=None, q=None, s=None, extra_args=None):
        return self.model(z, pos, batch=batch, q=q, s=s, extra_args=extra_args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            extra_args = batch.to_dict()
            for a in ('y', 'neg_dy', 'z', 'pos', 'batch', 'q', 's'):
                if a in extra_args:
                    del extra_args[a]
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            y, neg_dy = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
                extra_args=extra_args
            )

        loss_y, loss_neg_dy = 0, 0
        if self.hparams.derivative:
            if "y" not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the negative derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                neg_dy = neg_dy + y.sum() * 0

            # negative derivative loss
            loss_neg_dy = loss_fn(neg_dy, batch.neg_dy)

            if stage in ["train", "val"] and self.hparams.ema_alpha_neg_dy < 1:
                if self.ema[stage + "_neg_dy"] is None:
                    self.ema[stage + "_neg_dy"] = loss_neg_dy.detach()
                # apply exponential smoothing over batches to neg_dy
                loss_neg_dy = (
                    self.hparams.ema_alpha_neg_dy * loss_neg_dy
                    + (1 - self.hparams.ema_alpha_neg_dy) * self.ema[stage + "_neg_dy"]
                )
                self.ema[stage + "_neg_dy"] = loss_neg_dy.detach()

            if self.hparams.neg_dy_weight > 0:
                self.losses[stage + "_neg_dy"].append(loss_neg_dy.detach())

        if "y" in batch:
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            # y loss
            loss_y = loss_fn(y, batch.y)

            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()

            if self.hparams.y_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        # total loss
        loss = loss_y * self.hparams.y_weight + loss_neg_dy * self.hparams.neg_dy_weight

        self.losses[stage].append(loss.detach())
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch + 1) % self.hparams.test_interval == 0
            )
            if should_reset:
                # reset validation dataloaders before and after testing epoch, which is faster
                # than skipping test validation steps by returning None
                self.trainer.reset_val_dataloader(self)

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_neg_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_neg_dy"] = torch.stack(
                    self.losses["train_neg_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_neg_dy"] = torch.stack(self.losses["val_neg_dy"]).mean()

                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                    result_dict["test_loss_neg_dy"] = torch.stack(
                        self.losses["test_neg_dy"]
                    ).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_neg_dy": [],
            "val_neg_dy": [],
            "test_neg_dy": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_neg_dy": None, "val_neg_dy": None}
