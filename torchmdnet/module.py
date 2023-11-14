from collections import defaultdict
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import local_response_norm, mse_loss, l1_loss
from torch import Tensor
from typing import Optional, Dict, Tuple

from lightning import LightningModule
from torchmdnet.models.model import create_model, load_model


class LNNP(LightningModule):
    """
    Lightning wrapper for the Neural Network Potentials in TorchMD-Net.
    """

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

        self.weights = {}
        for key in self.model.state_dict():
            if "_weight" in key:
                self.weights[key] = self.model.state_dict()[key]

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

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        return self.model(z, pos, batch=batch, q=q, s=s, extra_args=extra_args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, [mse_loss], "train")

    def validation_step(self, batch, batch_idx, *args):
        # If args is not empty the first (and only) element is the dataloader_idx
        # We want to test every number of epochs just for reporting, but this is not supported by Lightning.
        # Instead, we trick it by providing two validation dataloaders and interpreting the second one as test.
        # The dataloader takes care of sending the two sets only when the second one is needed.
        is_val = len(args) == 0 or (len(args) > 0 and args[0] == 0)
        if is_val:
            step_type = {"loss_fn_list": [l1_loss, mse_loss], "stage": "val"}
        else:
            step_type = {"loss_fn_list": [l1_loss], "stage": "test"}
        return self.step(batch, **step_type)

    def test_step(self, batch, batch_idx):
        return self.step(batch, [l1_loss], "test")

    def _compute_losses(self, outputs, batch, loss_fn, stage):
        """
        Compute the losses for each model output.

        Args:
          outputs: Dictionary of model outputs.
          batch: Batch of data.
          loss_fn: Loss function to compute.
          stage: Current training stage.

        Returns:
          losses: Dictionary of computed losses for each model output.
        """
        losses = {}
        loss_name = loss_fn.__name__
        for key in outputs:
            if key in batch:
                loss = loss_fn(outputs[key], getattr(batch, key))
                loss = self._update_loss_with_ema(stage, key, loss_name, loss)
                losses[key] = loss
            else:
                raise ValueError(f"Reference values for '{key}' are missing in the batch")
        return losses

    def _update_loss_with_ema(self, stage, type, loss_name, loss):
        # Update the loss using an exponential moving average when applicable
        # Args:
        #   stage: stage of the training (train, val, test)
        #   type: type of loss (y, neg_dy)
        #   loss_name: name of the loss function
        #   loss: loss value
        alpha = getattr(self.hparams, f"ema_alpha_{type}")
        if stage in ["train", "val"] and alpha < 1:
            ema = (
                self.ema[stage][type][loss_name]
                if loss_name in self.ema[stage][type]
                else loss.detach()
            )
            loss = alpha * loss + (1 - alpha) * ema
            self.ema[stage][type][loss_name] = loss.detach()
        return loss

    def step(self, batch, loss_fn_list, stage):
        # Run a forward pass and compute the loss for each loss function
        # If the batch contains the derivative, also compute the loss for the negative derivative
        # Args:
        #   batch: batch of data
        #   loss_fn_list: list of loss functions to compute and record (the last one is used for the total loss returned by this function)
        #   stage: stage of the training (train, val, test)
        # Returns:
        #   total_loss: sum of all losses (weighted by the loss weights) for the last loss function in the provided list
        assert len(loss_fn_list) > 0
        assert self.losses is not None
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            extra_args = batch.to_dict()
            for a in ("y", "neg_dy", "z", "pos", "batch", "q", "s"):
                if a in extra_args:
                    del extra_args[a]
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            outputs = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
                extra_args=extra_args,
            )
        # Raul 2023, I don't think this is needed anymore
        # if self.hparams.derivative and "energy" not in batch:
        #     # "use" both outputs of the model's forward function but discard the first
        #     # to only use the negative derivative and avoid 'Expected to have finished reduction
        #     # in the prior iteration before starting a new one.', which otherwise get's
        #     # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
        #     outputs["forces"] += outputs["energy"].sum() * 0
        for loss_fn in loss_fn_list:
            step_losses = self._compute_losses(outputs, batch, loss_fn, stage)

            loss_name = loss_fn.__name__
            total_loss = torch.tensor(0.0, device=self.device, dtype=batch.pos.dtype)
            for key, loss in step_losses.items():
                if key in self.weights and self.weights[key] > 0:
                    self.losses[stage][key][loss_name].append(loss.detach())
                    total_loss += loss * self.weights[key]
            self.losses[stage]["total"][loss_name].append(total_loss.detach())
        return total_loss

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

    def _get_mean_loss_dict_for_type(self, type):
        # Returns a list with the mean loss for each loss_fn for each stage (train, val, test)
        # Parameters:
        # type: either y, neg_dy or total
        # Returns:
        # A dict with an entry for each stage (train, val, test) with the mean loss for each loss_fn (e.g. mse_loss)
        # The key for each entry is "stage_type_loss_fn"
        assert self.losses is not None
        mean_losses = {}
        for stage in ["train", "val", "test"]:
            for loss_fn_name in self.losses[stage][type].keys():
                mean_losses[stage + "_" + type + "_" + loss_fn_name] = torch.stack(
                    self.losses[stage][type][loss_fn_name]
                ).mean()
        return mean_losses

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
            result_dict.update(self._get_mean_loss_dict_for_type("total"))
            for i in self.weights.keys():
                result_dict.update(self._get_mean_loss_dict_for_type(i))
            self.log_dict(result_dict, sync_dist=True)

        self._reset_losses_dict()

    def on_test_epoch_end(self):
        # Log all test losses
        if not self.trainer.sanity_checking:
            result_dict = {}
            result_dict.update(self._get_mean_loss_dict_for_type("total"))
            for i in self.weights.keys():
                result_dict.update(self._get_mean_loss_dict_for_type(i))
            # Get only test entries
            result_dict = {k: v for k, v in result_dict.items() if k.startswith("test")}
            self.log_dict(result_dict, sync_dist=True)

    def _reset_losses_dict(self):
        # Losses has an entry for each stage in ["train", "val", "test"]
        # Each entry has an entry with "total" and each of the weights keys (e.g. y, neg_dy)
        # Each of these entries has an entry for each loss_fn (e.g. mse_loss)
        # The loss_fn values are not known in advance
        self.losses = {}
        for stage in ["train", "val", "test"]:
            self.losses[stage] = {}
            types = ["total"] + list(self.weights.keys())
            for loss_type in types:
                self.losses[stage][loss_type] = defaultdict(list)

    def _reset_ema_dict(self):
        self.ema = {}
        for stage in ["train", "val"]:
            self.ema[stage] = {}
            types = ["total"] + list(self.weights.keys())
            for loss_type in types:
                self.ema[stage][loss_type] = {}
