import os
import pytest

ONNXSCRIPT_AVAILABLE = True
try:
    import onnxscript
except ImportError:
    ONNXSCRIPT_AVAILABLE = False


curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skipif(not ONNXSCRIPT_AVAILABLE, reason="onnxscript not available")
def test_onnx_export(tmp_path):
    from torchmdnet.models.model import create_model
    from utils import load_example_args
    import torch as pt

    device = "cuda"  # "cuda" if pt.cuda.is_available() else "cpu"

    ben = {
        "z": pt.tensor(
            [6, 6, 6, 6, 6, 6, 6, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            dtype=pt.long,
            device=device,
        ),
        "pos": pt.tensor(
            [
                [-1.853, 14.311, 16.658],
                [-2.107, 15.653, 16.758],
                [-1.774, 16.341, 17.932],
                [-1.175, 15.662, 19.005],
                [-0.914, 14.295, 18.885],
                [-1.257, 13.634, 17.708],
                [-2.193, 13.627, 15.496],
                [-2.797, 14.235, 14.491],
                [-1.762, 12.391, 15.309],
                [-2.571735, 16.189823, 15.917855],
                [-1.9827466, 17.41793, 18.013515],
                [-0.91450024, 16.199741, 19.928564],
                [-0.44179577, 13.745465, 19.71267],
                [-1.0511663, 12.557756, 17.61139],
                [-3.038939, 13.707173, 13.640257],
                [-3.0260794, 15.236963, 14.558013],
                [-2.0127733, 11.882723, 14.448961],
                [-1.175496, 11.936485, 16.023375],
            ],
            dtype=pt.float32,
            device=device,
            requires_grad=True,
        ),
        "batch": pt.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=pt.long,
            device=device,
        ),
        "box": pt.tensor(
            [
                [100, 0.0, 0.0],
                [0.0, 100, 0.0],
                [0.0, 0.0, 100],
            ],
            dtype=pt.float32,
            device=device,
        ),
        "q": pt.tensor([1], dtype=pt.long, device=device),
    }
    # Water example
    example = {
        "z": pt.tensor([8, 1, 1], dtype=pt.long, device=device, requires_grad=False),
        "pos": pt.tensor(
            [
                [60.243, 56.013, 55.451],
                [61.061, 56.406, 55.263],
                [60.513, 55.071, 55.526],
            ],
            dtype=pt.float32,
            device=device,
            requires_grad=True,
        ),
        "batch": pt.tensor(
            [0, 0, 0],
            dtype=pt.long,
            device=device,
            requires_grad=False,
        ),
        "box": pt.tensor(
            [
                [100, 0.0, 0.0],
                [0.0, 100, 0.0],
                [0.0, 0.0, 100],
            ],
            dtype=pt.float32,
            device=device,
            requires_grad=False,
        ),
        "q": pt.tensor([0], dtype=pt.long, device=device, requires_grad=False),
    }

    model = create_model(
        load_example_args(
            "tensornet",
            prior_model=None,
            precision=32,
            derivative=True,
            static_shapes=False,
            onnx_export=True,
        )
    )

    model.to(device)
    model.eval()
    out = model(**example)
    print(out)

    pt.onnx.export(
        model,  # model to export
        (
            example["z"],
            example["pos"],
            example["batch"],
            example["box"],
            example["q"],
        ),  # inputs of the model,
        os.path.join(tmp_path, "my_model.onnx"),  # filename of the ONNX model
        input_names=[
            "z",
            "pos",
            "batch",
            "box",
            "q",
        ],  # Rename inputs for the ONNX model
        output_names=["energy", "forces"],
        dynamic_axes={
            "z": {0: "atoms"},
            "pos": {0: "atoms"},
            "batch": {0: "atoms"},
            # "energy": {0: "batch"},
            "forces": {0: "atoms"},
        },
        dynamo=False,  # True or False to select the exporter to use
        report=True,
        opset_version=20,
    )

    # Test the exported ONNX model
    import onnx
    import onnxruntime as ort

    model_path = os.path.join(tmp_path, "my_model.onnx")
    session = ort.InferenceSession(model_path)
    inputs = {
        "z": example["z"].cpu().numpy(),
        "pos": example["pos"].detach().cpu().numpy(),
        "batch": example["batch"].cpu().numpy(),
        # "box": example["box"].cpu().numpy(),
        "q": example["q"].cpu().numpy(),
    }
    outputs = session.run(None, inputs)
    print(outputs)
