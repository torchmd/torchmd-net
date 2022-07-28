from pytest import mark
import yaml
import glob
from os.path import dirname, join
from torchmdnet.models.model import create_model
from torchmdnet import priors
from utils import DummyDataset, create_example_batch


@mark.parametrize(
    "fname", glob.glob(join(dirname(dirname(__file__)), "examples", "*.yaml"))
)
def test_example_yamls(fname):
    with open(fname, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    prior = None
    if args["prior_model"] is not None:
        dataset = DummyDataset(has_atomref=True)
        prior = getattr(priors, args["prior_model"])(dataset=dataset)

    model = create_model(args, prior_model=prior)

    z, pos, batch = create_example_batch()
    model(z, pos, batch)
    model(z, pos, batch, q=None, s=None)
