import torch as pt
from tempfile import TemporaryDirectory
from torch_geometric.loader import DataLoader
from torchmdnet.datasets import S66X8


# Test smallet dataset from COMP6
def test_dataset_s66x8():

    with TemporaryDirectory() as root:
        data_set = S66X8(root)

        assert len(data_set) == 528

        sample = data_set[42]
        assert pt.equal(sample.z, pt.tensor([8, 1, 6, 1, 1, 1, 7, 1, 1, 6, 1, 1, 1]))
        assert pt.allclose(
            sample.pos,
            pt.tensor(
                [
                    [-1.5489e00, -7.4912e-01, -1.2941e-02],
                    [-5.7664e-01, -7.8385e-01, -4.2315e-03],
                    [-1.9064e00, 6.1374e-01, 7.7022e-03],
                    [-2.9922e00, 6.7289e-01, -3.7969e-04],
                    [-1.5522e00, 1.1299e00, 9.0522e-01],
                    [-1.5373e00, 1.1606e00, -8.6531e-01],
                    [1.3905e00, -6.9185e-01, 1.0001e-02],
                    [1.7599e00, -1.1928e00, -7.8803e-01],
                    [1.7536e00, -1.1699e00, 8.2477e-01],
                    [1.8809e00, 6.9018e-01, -7.7681e-03],
                    [1.4901e00, 1.2167e00, 8.5957e-01],
                    [2.9685e00, 7.8712e-01, -4.9738e-03],
                    [1.4969e00, 1.1917e00, -8.9271e-01],
                ]
            ),
            atol=1e-4,
        )
        assert pt.allclose(sample.y, pt.tensor([[-47.5919]]))
        assert pt.allclose(
            sample.neg_dy,
            pt.tensor(
                [
                    [0.2739, -0.2190, -0.0012],
                    [-0.2938, 0.0556, -0.0023],
                    [-0.1230, 0.4893, 0.0061],
                    [0.2537, -0.0083, 0.0024],
                    [-0.1036, -0.1406, -0.2298],
                    [-0.1077, -0.1492, 0.2241],
                    [0.0509, -0.2316, 0.0030],
                    [-0.0080, 0.0953, 0.1822],
                    [-0.0064, 0.0900, -0.1851],
                    [0.1478, 0.2256, -0.0017],
                    [0.0913, -0.1069, -0.1955],
                    [-0.2650, 0.0011, -0.0012],
                    [0.0900, -0.1013, 0.1990],
                ]
            ),
            atol=1e-4,
        )

        data_loader = DataLoader(dataset=data_set, batch_size=32, num_workers=2)
        for batch in data_loader:
            pass
