# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import io
from pathlib import Path
import tarfile

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import pytest
import torch
from torch_geometric.loader import DataLoader
import yaml

from torchmdnet.data import DataModule
from torchmdnet.datasets.maceoff import MACEOFF
from torchmdnet.models.model import create_prior_models, load_model
from torchmdnet.module import EnergyRefRemover, LNNP
from torchmdnet.scripts.train import get_argparse

# Configurations 0, 1, 5, 6, 8, 12 and 63 from the MIT-licensed training data by
# Moore et al., https://doi.org/10.17863/CAM.107498, cover all ten elements.
# These offsets sum the published MACE-OFF23 E0 values with decimal arithmetic.
MOLECULAR_REFERENCES = torch.tensor(
    [
        -8461.688754979813088,
        -171027.036800161518852,
        -26943.220350106981744,
        -39291.608891759186714,
        -21449.260092551987744,
        -46822.222982158050918,
        -73383.874629915429006,
    ],
    dtype=torch.float64,
).view(-1, 1)


@pytest.fixture
def maceoff_root(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "maceoff" / "1.0"
    raw_dir.mkdir(parents=True)
    contents = Path(__file__).with_name("maceoff_sample.xyz").read_bytes()
    with tarfile.open(
        raw_dir / "train_large_neut_no_bad_clean.tar.gz", "w:gz"
    ) as archive:
        member = tarfile.TarInfo("train_large_neut_no_bad_clean.xyz")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    return tmp_path / "data"


@pytest.mark.parametrize("max_z", [54, 128])
def test_maceoff_reference_energy_lifecycle(maceoff_root, max_z):
    dataset = MACEOFF(str(maceoff_root))
    original = next(iter(DataLoader(dataset, batch_size=7)))
    processed = {
        Path(path): Path(path).stat().st_mtime_ns for path in dataset.processed_paths
    }
    for source in (dataset, MACEOFF(str(maceoff_root))):
        references = source.get_atomref(max_z)
        if references.shape != (max_z, 1):
            pytest.fail(
                "The reference table did not preserve the requested atomic-number range"
            )
        batch = next(iter(DataLoader(source, batch_size=7)))
        shifted = EnergyRefRemover(references)(batch.clone())
        expected = original.y - MOLECULAR_REFERENCES
        if not torch.allclose(shifted.y, expected, atol=1e-8, rtol=0):
            pytest.fail(
                "MACEOFF delta targets disagree with the published reference energies"
            )
        if not torch.equal(shifted.neg_dy, original.neg_dy):
            pytest.fail("Removing constant reference energies changed the force labels")
        if not torch.equal(batch.y, original.y) or not torch.equal(
            batch.pos, original.pos
        ):
            pytest.fail(
                "Reference-energy removal changed the stored total energies or coordinates"
            )
    if any(path.stat().st_mtime_ns != mtime for path, mtime in processed.items()):
        pytest.fail("Reopening MACEOFF reprocessed its cached molecular data")


def test_maceoff_delta_training_and_checkpoint(maceoff_root, tmp_path):
    torch.set_num_threads(1)
    args = vars(get_argparse().parse_args([]))
    recipe = Path(__file__).parents[1] / "examples" / "TensorNet-MACEOFF.yaml"
    args.update(yaml.safe_load(recipe.read_text()))
    args.update(
        dataset_root=str(maceoff_root),
        log_dir=str(tmp_path),
        precision=64,
        remove_ref_energy=True,
        prior_model=[{"Atomref": {"enable": False}}],
        embedding_dimension=4,
        num_layers=1,
        num_rbf=4,
        batch_size=2,
        inference_batch_size=2,
        num_workers=0,
        train_size=4,
        val_size=2,
        test_size=1,
        test_interval=-1,
        lr_warmup_steps=0,
    )
    data = DataModule(args)
    data.setup("fit")
    priors = create_prior_models(args, data.dataset)
    args["prior_args"] = [prior.get_init_args() for prior in priors]
    model = LNNP(args, prior_model=priors)
    original = next(iter(DataLoader(data.dataset, batch_size=7)))
    transformed = model.data_transform(original.clone())
    if not torch.allclose(
        transformed.y, original.y - MOLECULAR_REFERENCES, atol=1e-8, rtol=0
    ):
        pytest.fail(
            "The training consumer did not receive reference-subtracted energies"
        )
    if not torch.equal(transformed.neg_dy, original.neg_dy.to(torch.float64)):
        pytest.fail("The delta-learning transform changed the force labels")
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision=64,
        max_epochs=1,
        callbacks=[ModelCheckpoint(dirpath=tmp_path)],
        logger=CSVLogger(str(tmp_path), name="", version=""),
        inference_mode=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, data)
    checkpoint = tmp_path / "delta.ckpt"
    trainer.save_checkpoint(checkpoint)
    if trainer.global_step != 2 or not torch.isfinite(
        trainer.callback_metrics["val_total_mse_loss"]
    ):
        pytest.fail("Delta training did not complete two finite optimizer steps")
    delta_model = load_model(str(checkpoint))
    total_model = load_model(str(checkpoint), remove_ref_energy=False)
    inputs = (original.z, original.pos.to(torch.float64), original.batch)
    delta_energy, delta_force = delta_model(*inputs)
    total_energy, total_force = total_model(*inputs)
    if not torch.allclose(
        total_energy - delta_energy, MOLECULAR_REFERENCES, atol=1e-8, rtol=0
    ):
        pytest.fail(
            "Checkpoint inference did not restore the published atomic reference energies"
        )
    if not torch.equal(total_force, delta_force):
        pytest.fail(
            "Restoring constant reference energies changed the predicted forces"
        )


def test_maceoff_reference_standardization(maceoff_root, tmp_path):
    args = vars(get_argparse().parse_args([]))
    args.update(
        dataset="MACEOFF",
        dataset_root=str(maceoff_root),
        log_dir=str(tmp_path),
        prior_model="Atomref",
        standardize=True,
        train_size=7,
        val_size=0,
        test_size=0,
        inference_batch_size=7,
        num_workers=0,
    )
    data = DataModule(args)
    data.setup("fit")
    batch = next(iter(DataLoader(data.dataset, batch_size=7)))
    residuals = batch.y - MOLECULAR_REFERENCES
    if not torch.allclose(data.mean, residuals.mean(dim=0), atol=1e-8, rtol=0):
        pytest.fail(
            "Atomref standardization did not center the reference-subtracted energies"
        )
    if not torch.allclose(data.std, residuals.std(dim=0), atol=1e-8, rtol=0):
        pytest.fail("Atomref standardization did not use the residual energy scale")
