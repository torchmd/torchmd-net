# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from collections import defaultdict
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import local_response_norm
from torch import Tensor
from typing import Optional, Dict, Tuple
from lightning import LightningModule
from torchmdnet.models.model import create_model, load_model
from torchmdnet.models.utils import dtype_mapping
from torchmdnet.loss import l1_loss, loss_class_mapping
import torch_geometric.transforms as T


class FloatCastDatasetWrapper(T.BaseTransform):
    """A transform that casts all floating point tensors to a given dtype.
    tensors to a given dtype.
    """

    def __init__(self, dtype=torch.float64):
        super(FloatCastDatasetWrapper, self).__init__()
        self._dtype = dtype

    def __call__(self, data):
        for key, value in data:
            if torch.is_tensor(value) and torch.is_floating_point(value):
                setattr(data, key, value.to(self._dtype))
        return data


class EnergyRefRemover(T.BaseTransform):
    """A transform that removes the atom reference energy from the energy of a
    dataset.
    """

    def __init__(self, atomref):
        super(EnergyRefRemover, self).__init__()
        self._atomref = atomref

    def __call__(self, data):
        self._atomref = self._atomref.to(data.z.device).type(data.y.dtype)
        if "y" in data:
            data.y.index_add_(0, data.batch, -self._atomref[data.z])
        return data


# This wrapper is here in order to permit Lightning to serialize the loss function.
class LossFunction:
    def __init__(self, loss_fn, extra_args=None):
        self.loss_fn = loss_fn
        self.extra_args = extra_args
        if self.extra_args is None:
            self.extra_args = {}

    def __call__(self, x, batch):
        return self.loss_fn(x, batch, **self.extra_args)


class LNNP(LightningModule):
    """
    Lightning wrapper for the Neural Network Potentials in TorchMD-Net.

    Args:
        hparams (dict): A dictionary containing the hyperparameters of the model.
        prior_model (torchmdnet.priors.BasePrior): A prior model to use in the model.
        mean (torch.Tensor, optional): The mean of the dataset to normalize the input.
        std (torch.Tensor, optional): The standard deviation of the dataset to normalize the input.
    """

    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        if "charge" not in hparams:
            hparams["charge"] = False
        if "spin" not in hparams:
            hparams["spin"] = False
        if "train_loss" not in hparams:
            hparams["train_loss"] = "mse_loss"
        if "train_loss_arg" not in hparams:
            hparams["train_loss_arg"] = {}
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

        self.data_transform = FloatCastDatasetWrapper(
            dtype_mapping[self.hparams.precision]
        )
        if self.hparams.remove_ref_energy:
            self.data_transform = T.Compose(
                [
                    EnergyRefRemover(self.model.prior_model[-1].initial_atomref),
                    self.data_transform,
                ]
            )

        if self.hparams.train_loss not in loss_class_mapping:
            raise ValueError(
                f"Training loss {self.hparams.train_loss} not supported. Supported losses are {list(loss_class_mapping.keys())}"
            )

        self.train_loss_fn = LossFunction(
            loss_class_mapping[self.hparams.train_loss],
            self.hparams.train_loss_arg,
        )

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
        lr_metric = getattr(self.hparams, "lr_metric", "val")
        monitor = f"{lr_metric}_total_{self.hparams.train_loss}"
        lr_scheduler = {
            "scheduler": scheduler,
            "strict": True,
            "monitor": monitor,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.model(z, pos, batch=batch, box=box, q=q, s=s, extra_args=extra_args)

    def training_step(self, batch, batch_idx):
        return self.step(
            batch, [(self.hparams.train_loss, self.train_loss_fn)], "train"
        )

    def validation_step(self, batch, batch_idx, *args):
        # If args is not empty the first (and only) element is the dataloader_idx
        # We want to test every number of epochs just for reporting, but this is not supported by Lightning.
        # Instead, we trick it by providing two validation dataloaders and interpreting the second one as test.
        # The dataloader takes care of sending the two sets only when the second one is needed.
        is_val = len(args) == 0 or (len(args) > 0 and args[0] == 0)
        if is_val:
            step_type = {
                "loss_fn_list": [
                    ("l1_loss", l1_loss),
                    (self.hparams.train_loss, self.train_loss_fn),
                ],
                "stage": "val",
            }
        else:
            step_type = {"loss_fn_list": [("l1_loss", l1_loss)], "stage": "test"}
        return self.step(batch, **step_type)

    def test_step(self, batch, batch_idx):
        return self.step(batch, [("l1_loss", l1_loss)], "test")

    def predict_step(self, batch, batch_idx):
        batch = self.data_transform(batch)

        with torch.set_grad_enabled(self.hparams.derivative):
            extra_args = batch.to_dict()
            for a in ("y", "neg_dy", "z", "pos", "batch", "box", "q", "s"):
                if a in extra_args:
                    del extra_args[a]
            return self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
                extra_args=extra_args,
            )

    def _compute_losses(self, y, neg_y, batch, loss_fn, loss_name, stage):
        # Compute the loss for the predicted value and the negative derivative (if available)
        # Args:
        #   y: predicted value
        #   neg_y: predicted negative derivative
        #   batch: batch of data
        #   loss_fn: The loss function to compute
        #   loss_name: The name of the loss function
        # Returns:
        #   loss_y: loss for the predicted value
        #   loss_neg_y: loss for the predicted negative derivative
        loss_y, loss_neg_y = torch.tensor(0.0, device=self.device), torch.tensor(
            0.0, device=self.device
        )
        if self.hparams.derivative and "neg_dy" in batch:
            loss_neg_y = loss_fn(neg_y, batch.neg_dy)
            loss_neg_y = self._update_loss_with_ema(
                stage, "neg_dy", loss_name, loss_neg_y
            )
        if "y" in batch:
            loss_y = loss_fn(y, batch.y)
            loss_y = self._update_loss_with_ema(stage, "y", loss_name, loss_y)
        return {"y": loss_y, "neg_dy": loss_neg_y}

    def _update_loss_with_ema(self, stage, type, loss_name, loss):
        # Update the loss using an exponential moving average when applicable
        # Args:
        #   stage: stage of the training (train, val, test)
        #   type: type of loss (y, neg_dy)
        #   loss_name: name of the loss function
        #   loss: loss value
        alpha = getattr(self.hparams, f"ema_alpha_{type}")
        if stage in ["train", "val"] and alpha < 1 and alpha > 0:
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
        batch = self.data_transform(batch)
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            extra_args = batch.to_dict()
            for a in ("y", "neg_dy", "z", "pos", "batch", "box", "q", "s"):
                if a in extra_args:
                    del extra_args[a]
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            y, neg_dy = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
                extra_args=extra_args,
            )
        if self.hparams.derivative and "y" not in batch:
            # "use" both outputs of the model's forward function but discard the first
            # to only use the negative derivative and avoid 'Expected to have finished reduction
            # in the prior iteration before starting a new one.', which otherwise get's
            # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
            neg_dy = neg_dy + y.sum() * 0
        if "y" in batch and batch.y.ndim == 1:
            batch.y = batch.y.unsqueeze(1)
        for loss_name, loss_fn in loss_fn_list:
            step_losses = self._compute_losses(
                y, neg_dy, batch, loss_fn, loss_name, stage
            )
            if self.hparams.neg_dy_weight > 0:
                self.losses[stage]["neg_dy"][loss_name].append(
                    step_losses["neg_dy"].detach()
                )
            if self.hparams.y_weight > 0:
                self.losses[stage]["y"][loss_name].append(step_losses["y"].detach())
            total_loss = (
                step_losses["y"] * self.hparams.y_weight
                + step_losses["neg_dy"] * self.hparams.neg_dy_weight
            )
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
            result_dict.update(self._get_mean_loss_dict_for_type("y"))
            result_dict.update(self._get_mean_loss_dict_for_type("neg_dy"))
            self.log_dict(result_dict, sync_dist=True)

        self._reset_losses_dict()

    def on_test_epoch_end(self):
        # Log all test losses
        if not self.trainer.sanity_checking:
            result_dict = {}
            result_dict.update(self._get_mean_loss_dict_for_type("total"))
            result_dict.update(self._get_mean_loss_dict_for_type("y"))
            result_dict.update(self._get_mean_loss_dict_for_type("neg_dy"))
            # Get only test entries
            result_dict = {k: v for k, v in result_dict.items() if k.startswith("test")}
            self.log_dict(result_dict, sync_dist=True)

    def on_train_epoch_end(self):
        # Log all train losses
        if not self.trainer.sanity_checking:
            result_dict = {}
            result_dict.update(self._get_mean_loss_dict_for_type("total"))
            result_dict.update(self._get_mean_loss_dict_for_type("y"))
            result_dict.update(self._get_mean_loss_dict_for_type("neg_dy"))
            # Get only train entries
            result_dict = {
                k: v for k, v in result_dict.items() if k.startswith("train")
            }
            self.log_dict(result_dict, sync_dist=True)

    def _reset_losses_dict(self):
        # Losses has an entry for each stage in ["train", "val", "test"]
        # Each entry has an entry with "total", "y" and "neg_dy"
        # Each of these entries has an entry for each loss_fn (e.g. mse_loss)
        # The loss_fn values are not known in advance
        self.losses = {}
        for stage in ["train", "val", "test"]:
            self.losses[stage] = {}
            for loss_type in ["total", "y", "neg_dy"]:
                self.losses[stage][loss_type] = defaultdict(list)

    def _reset_ema_dict(self):
        self.ema = {}
        for stage in ["train", "val"]:
            self.ema[stage] = {}
            for loss_type in ["y", "neg_dy"]:
                self.ema[stage][loss_type] = {}
