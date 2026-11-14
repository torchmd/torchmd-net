# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import itertools
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml


def run_training(arguments, cwd):
    result = subprocess.run(
        [sys.executable, "-m", "torchmdnet.scripts.train", *map(str, arguments)],
        cwd=cwd,
        env={**os.environ, "OMP_NUM_THREADS": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        pytest.fail(result.stdout + result.stderr)
    return result


@pytest.fixture(scope="module")
def training_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    positions = np.zeros((8, 2, 3), dtype=np.float32)
    positions[:, 1, 0] = np.linspace(0.7, 1.2, 8)
    np.save(root / "coords.npy", positions)
    np.save(root / "types.npy", np.array([1, 1]))
    np.save(root / "energy.npy", ((positions[:, 1, 0] - 1) ** 2).reshape(-1, 1))
    config = dict(
        dataset="Custom",
        coord_files=str(root / "coords.npy"),
        embed_files=str(root / "types.npy"),
        energy_files=str(root / "energy.npy"),
        model="tensornet",
        embedding_dimension=4,
        num_layers=1,
        num_rbf=4,
        batch_size=2,
        inference_batch_size=2,
        train_size=4,
        val_size=2,
        test_size=2,
        num_workers=0,
        num_epochs=1,
        save_interval=1,
        ngpus=1,
        derivative=False,
        neg_dy_weight=0,
        log_dir=str(root / "seed"),
    )
    (root / "seed.yaml").write_text(yaml.safe_dump(config))
    run_training(["--conf", root / "seed.yaml"], root)
    return root / "seed"


@pytest.mark.parametrize(
    "order", list(itertools.permutations(("checkpoint", "config", "cli")))
)
def test_resume_training_precedence(training_checkpoint, tmp_path, order):
    previous = tmp_path / "previous"
    shutil.copytree(training_checkpoint, previous)
    hparams_path = previous / "hparams.yaml"
    hparams = yaml.safe_load(hparams_path.read_text())
    hparams["log_dir"] = str(previous)
    hparams_path.write_text(yaml.safe_dump(hparams))
    original_files = {path.name: path.read_bytes() for path in previous.iterdir()}
    checkpoint = next(previous.glob("*.ckpt"))
    config = dict(
        log_dir=str(tmp_path / "config-output"),
        batch_size=3,
        num_epochs=2,
        train_size=2,
        y_weight=2.5,
        lr_warmup_steps=4,
    )
    config_path = tmp_path / "resume.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "selected output"
    groups = {
        "checkpoint": ["--load-model", checkpoint],
        "config": ["--conf", config_path],
        "cli": [
            "--log-dir",
            output,
            "--batch-size",
            "1",
            "--lr-warmup-steps",
            "0",
            "--train-size",
            "None",
        ],
    }
    run_training([value for group in order for value in groups[group]], tmp_path)
    if not (output / "input.yaml").is_file():
        pytest.fail("The explicit output directory was replaced by inherited settings")
    expected = dict(
        log_dir=str(output),
        batch_size=1,
        num_epochs=2,
        train_size=None,
        y_weight=2.5,
        lr_warmup_steps=0,
        redirect=False,
        reset_trainer=False,
        prior_model=None,
        model="tensornet",
        embedding_dimension=4,
        load_model=str(checkpoint),
    )
    for filename in ("input.yaml", "hparams.yaml"):
        saved = yaml.safe_load((output / filename).read_text())
        for name, value in expected.items():
            if saved[name] != value:
                pytest.fail(
                    f"{filename}: {name} was {saved[name]!r}, expected {value!r}"
                )
    with np.load(output / "splits.npz") as splits:
        if [len(splits[key]) for key in ("idx_train", "idx_val", "idx_test")] != [
            4,
            2,
            2,
        ]:
            pytest.fail("Explicit train-size None did not reach the dataset split")
    restored = torch.load(
        next(output.glob("*.ckpt")), map_location="cpu", weights_only=False
    )
    initial = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if restored["epoch"] != 1 or restored["global_step"] != initial["global_step"] + 4:
        pytest.fail("The trainer did not resume its state with the selected batch size")
    if all(
        torch.equal(value, restored["state_dict"][name])
        for name, value in initial["state_dict"].items()
    ):
        pytest.fail("Resumed training did not update the model")
    if not (output / "metrics.csv").is_file() or (output / "log").exists():
        pytest.fail("The selected logging settings did not reach the training consumer")
    if (tmp_path / "config-output").exists():
        pytest.fail(
            "The config output directory was used despite the explicit CLI value"
        )
    for name, contents in original_files.items():
        if (previous / name).read_bytes() != contents:
            pytest.fail(
                f"Resuming into another directory overwrote {name} in the old run"
            )


@pytest.mark.parametrize(
    "source,sidecar",
    [("config", True), ("config", False), ("cli", False), ("cli", True)],
)
def test_resume_checkpoint_selection(training_checkpoint, tmp_path, source, sidecar):
    previous = tmp_path / "previous"
    previous.mkdir()
    checkpoint = previous / "seed.ckpt"
    shutil.copyfile(next(training_checkpoint.glob("*.ckpt")), checkpoint)
    output = tmp_path / "resumed"
    if sidecar:
        shutil.copyfile(training_checkpoint / "hparams.yaml", previous / "hparams.yaml")
        config = {}
    else:
        config = yaml.safe_load((training_checkpoint / "input.yaml").read_text())
    checkpoint_arg = str(checkpoint.relative_to(tmp_path))
    config.update(load_model=checkpoint_arg, num_epochs=2, log_dir=str(output))
    arguments = ["--conf", "resume.yaml"]
    if source == "cli":
        config["load_model"] = "unused.ckpt"
        arguments = ["--load-model", checkpoint_arg, "--redirect", "True", *arguments]
        if sidecar:
            config.update(redirect=True, reset_trainer=True, prior_model=["Atomref"])
            arguments = [
                "--load-model",
                checkpoint_arg,
                "--redirect",
                "False",
                "--reset-trainer",
                "False",
                "--prior-model",
                "--conf",
                "resume.yaml",
            ]
    (tmp_path / "resume.yaml").write_text(yaml.safe_dump(config))
    result = run_training(arguments, tmp_path)
    resumed = torch.load(
        next(output.glob("*.ckpt")), map_location="cpu", weights_only=False
    )
    if resumed["hyper_parameters"]["load_model"] != checkpoint_arg:
        pytest.fail("The checkpoint selected in the config was not passed to the model")
    if resumed["epoch"] != 1 or resumed["global_step"] != 4:
        pytest.fail("The checkpoint selected in the config did not resume training")
    logs = result.stderr
    if source == "cli" and not sidecar:
        logs += (output / "log").read_text()
    if source == "cli" and sidecar:
        saved = resumed["hyper_parameters"]
        if saved["redirect"] or saved["reset_trainer"] or saved["prior_model"] != []:
            pytest.fail(
                "Explicitly disabled options did not override the configuration"
            )
        if (output / "log").exists():
            pytest.fail("Explicit redirect False still redirected training output")
    if "Restored all states from the checkpoint" not in logs:
        pytest.fail("Training did not restore the checkpoint state")


def test_repeated_training_configuration(training_checkpoint, tmp_path):
    from torchmdnet.data import DataModule
    from torchmdnet.scripts.train import get_args

    original_argv = sys.argv
    try:
        for index, batch_size in enumerate((1, 2)):
            output = tmp_path / str(index)
            sys.argv = [
                "torchmd-train",
                "--conf",
                str(training_checkpoint / "input.yaml"),
                "--log-dir",
                str(output),
            ]
            if index == 0:
                sys.argv.extend(["--batch-size", "1"])
            args = get_args()
            data = DataModule(args)
            data.setup("fit")
            batch = next(iter(data.train_dataloader()))
            if batch.num_graphs != batch_size:
                pytest.fail("A previous parse changed the next training batch size")
            saved = yaml.safe_load((output / "input.yaml").read_text())
            if saved["batch_size"] != batch_size:
                pytest.fail(
                    "Persisted settings disagree with the selected training batch"
                )
        output = tmp_path / "defaults"
        sys.argv = ["torchmd-train", "--log-dir", str(output)]
        defaults = get_args()
        if (
            defaults.batch_size,
            defaults.inference_batch_size,
            defaults.load_model,
        ) != (32, 32, None):
            pytest.fail("Earlier parses changed the built-in defaults")
    finally:
        sys.argv = original_argv


@pytest.mark.parametrize("contents", ["unknown_option: 1\n", "[]\n", ""])
def test_invalid_configuration_leaves_output_untouched(tmp_path, contents):
    config = tmp_path / "invalid.yaml"
    config.write_text(contents)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchmdnet.scripts.train",
            "--log-dir",
            str(output),
            "--conf",
            str(config),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0 or str(config) not in result.stderr:
        pytest.fail("Invalid configuration was not rejected with its source path")
    if output.exists():
        pytest.fail("Invalid configuration created training output")


@pytest.fixture(scope="module")
def prior_training_checkpoint(training_checkpoint, tmp_path_factory):
    root = tmp_path_factory.mktemp("prior-training")
    config = yaml.safe_load((training_checkpoint / "input.yaml").read_text())
    config.update(
        log_dir=str(root / "seed"),
        prior_model={
            "ZBL": dict(
                cutoff_distance=0.8,
                max_num_neighbors=16,
                atomic_number=[0, 1],
                distance_scale=1e-10,
                energy_scale=1.602176634e-19,
            )
        },
    )
    (root / "seed.yaml").write_text(yaml.safe_dump(config))
    run_training(["--conf", "seed.yaml"], root)
    return root / "seed"


@pytest.mark.parametrize(
    "definition",
    [
        "inline",
        "inline-list",
        "legacy-only",
        "same-source",
        "same-source-inline",
        "args-only",
    ],
)
def test_resume_prior_configuration(prior_training_checkpoint, tmp_path, definition):
    from torchmdnet.data import DataModule
    from torchmdnet.models.model import load_model

    previous = tmp_path / "previous"
    shutil.copytree(prior_training_checkpoint, previous)
    hparams_path = previous / "hparams.yaml"
    hparams = yaml.safe_load(hparams_path.read_text())
    hparams["prior_model"] = "ZBL"
    hparams_path.write_text(yaml.safe_dump(hparams))
    prior_args = {**hparams["prior_args"][0], "cutoff_distance": 1.3}
    output = tmp_path / "resumed"
    config = dict(
        load_model=str(next(previous.glob("*.ckpt"))),
        num_epochs=2,
        log_dir=str(output),
    )
    if definition == "inline":
        config["prior_model"] = {"ZBL": prior_args}
    elif definition == "inline-list":
        config["prior_model"] = [{"ZBL": prior_args}]
    elif definition == "same-source":
        config.update(prior_model="ZBL", prior_args=prior_args)
    elif definition == "same-source-inline":
        config.update(
            prior_model={"ZBL": {**prior_args, "cutoff_distance": 1.1}},
            prior_args=[prior_args],
        )
    elif definition == "args-only":
        config["prior_args"] = [prior_args]
    (tmp_path / "resume.yaml").write_text(yaml.safe_dump(config))
    run_training(["--conf", "resume.yaml"], tmp_path)
    checkpoint = next(output.glob("*.ckpt"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_cutoff = 0.8 if definition == "legacy-only" else 1.3
    if saved["hyper_parameters"]["prior_args"][0]["cutoff_distance"] != expected_cutoff:
        pytest.fail("Training saved a lower-priority prior cutoff in the checkpoint")
    if saved["global_step"] != 4:
        pytest.fail("Prior configuration was not exercised through resumed training")
    data = DataModule.load_from_checkpoint(str(checkpoint))
    data.setup("fit")
    if len(data.idx_train) != 4:
        pytest.fail("Checkpoint reload did not preserve the training dataset")
    model = load_model(str(checkpoint))
    prior = model.prior_model[0]
    if prior.cutoff_distance != expected_cutoff:
        pytest.fail("Checkpoint reload did not preserve the selected prior cutoff")
    energy = prior.post_reduce(
        torch.zeros((1, 1)),
        torch.tensor([1, 1]),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.zeros(2, dtype=torch.long),
    ).item()
    if (expected_cutoff < 1.0 and energy != 0) or (
        expected_cutoff > 1.0 and energy <= 0
    ):
        pytest.fail("The reloaded ZBL prior did not apply the selected distance cutoff")


def test_resume_reordered_priors(prior_training_checkpoint, tmp_path):
    from torchmdnet.models.model import load_model

    config = yaml.safe_load((prior_training_checkpoint / "input.yaml").read_text())
    first = config["prior_model"]["ZBL"]
    second = {**first, "cutoff_distance": 1.3}
    config.update(
        log_dir=str(tmp_path / "seed"),
        prior_model=[{"ZBL": first}, {"ZBL": second}],
    )
    (tmp_path / "seed.yaml").write_text(yaml.safe_dump(config))
    run_training(["--conf", "seed.yaml"], tmp_path)
    output = tmp_path / "resumed"
    resume = dict(
        load_model=str(next((tmp_path / "seed").glob("*.ckpt"))),
        log_dir=str(output),
        num_epochs=2,
        prior_model=[{"ZBL": second}, {"ZBL": first}],
    )
    (tmp_path / "resume.yaml").write_text(yaml.safe_dump(resume))
    run_training(["--conf", "resume.yaml"], tmp_path)
    checkpoint = next(output.glob("*.ckpt"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cutoffs = [
        prior["cutoff_distance"] for prior in saved["hyper_parameters"]["prior_args"]
    ]
    if cutoffs != [1.3, 0.8] or saved["global_step"] != 4:
        pytest.fail(
            "Resumed training used positional arguments from the old prior order"
        )
    model = load_model(str(checkpoint))
    if [prior.cutoff_distance for prior in model.prior_model] != [1.3, 0.8]:
        pytest.fail("Checkpoint reload changed the selected prior order")


@pytest.mark.parametrize("selection", [[], ["D2"]])
def test_explicit_prior_selection_replaces_inherited_arguments(
    prior_training_checkpoint, tmp_path, selection
):
    config = yaml.safe_load((prior_training_checkpoint / "hparams.yaml").read_text())
    config.update(
        load_model=str(next(prior_training_checkpoint.glob("*.ckpt"))),
        log_dir=str(tmp_path / "resumed"),
        num_epochs=2,
    )
    (tmp_path / "resume.yaml").write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchmdnet.scripts.train",
            "--prior-model",
            *selection,
            "--conf",
            str(tmp_path / "resume.yaml"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    saved = yaml.safe_load((tmp_path / "resumed" / "input.yaml").read_text())
    if saved["prior_model"] != selection or "prior_args" in saved:
        pytest.fail(
            "Explicit prior selection retained arguments from the old definition"
        )
    # Changing the checkpoint architecture or omitting required prior arguments remains invalid.
    error = (
        "required positional arguments"
        if selection
        else "Unexpected key(s) in state_dict"
    )
    if result.returncode == 0 or error not in result.stderr:
        pytest.fail(
            "Invalid prior selection bypassed the existing model compatibility check"
        )
