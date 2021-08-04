import yaml
from os.path import dirname, join
import torch


def load_example_args(model_type, prior=True):
    with open(join(dirname(dirname(__file__)), "examples", "example.yaml"), "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args["model"] = model_type
    if not prior:
        args["prior_model"] = None
    return args


def create_example_batch(n_atoms=6, multiple_batches=True):
    zs = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)
    z = zs[torch.randint(0, len(zs), (n_atoms,))]

    pos = torch.randn(len(z), 3)

    batch = torch.zeros(len(z), dtype=torch.long)
    if multiple_batches:
        batch[len(batch) // 2 :] = 1
    return z, pos, batch
