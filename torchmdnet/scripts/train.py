import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from torchmdnet.module import LNNP
from torchmdnet.data import DataModule
from torchmdnet.models.model import create_prior_models
from torchmdnet.utils import save_argparse
from lightning_utilities.core.rank_zero import rank_zero_warn
from torchmdnet.options import get_argparse

import os
import sys
import logging

def get_args():
    args = get_argparse().parse_args()
    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    prior_models = create_prior_models(vars(args), data.dataset)
    args.prior_args = [p.get_init_args() for p in prior_models]

    # initialize lightning module
    model = LNNP(args, prior_model=prior_models, mean=data.mean, std=data.std)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor="val_total_mse_loss",
        save_top_k=10,  # -1 to save all
        every_n_epochs=args.save_interval,
        filename="epoch={epoch}-val_loss={val_total_mse_loss:.4f}-test_loss={test_total_l1_loss:.4f}",
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        "val_total_mse_loss", patience=args.early_stopping_patience
    )

    csv_logger = CSVLogger(args.log_dir, name="", version="")
    _logger = [csv_logger]
    if args.wandb_use:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            save_dir=args.log_dir,
            resume="must" if args.wandb_resume_from_id is not None else None,
            id=args.wandb_resume_from_id,
        )
        _logger.append(wandb_logger)

    if args.tensorboard_use:
        tb_logger = TensorBoardLogger(
            args.log_dir, name="tensorbord", version="", default_hp_metric=False
        )
        _logger.append(tb_logger)
    if args.test_interval > 0:
        rank_zero_warn(
            f"WARNING: Test set will be evaluated every {args.test_interval} epochs. This will slow down training."
        )

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=args.ngpus,
        num_nodes=args.num_nodes,
        default_root_dir=args.log_dir,
        callbacks=[early_stopping, checkpoint_callback],
        logger=_logger,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        inference_mode=False,
        # Test-during-training requires reloading the dataloaders every epoch
        reload_dataloaders_every_n_epochs=1 if args.test_interval > 0 else 0,
    )

    trainer.fit(model, data, ckpt_path=None if args.reset_trainer else args.load_model)

    # run test set after completing the fit
    model = LNNP.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer = pl.Trainer(
        logger=_logger,
        inference_mode=False,
        accelerator="auto",
        devices=args.ngpus,
        num_nodes=args.num_nodes,
    )
    trainer.test(model, data)


if __name__ == "__main__":
    main()
