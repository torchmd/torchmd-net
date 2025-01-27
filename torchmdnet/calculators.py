# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torchmdnet.models.model import load_model
import warnings

# dict of preset transforms
tranforms = {
    "eV/A -> kcal/mol/A": lambda energy, forces: (
        energy * 23.0609,
        forces * 23.0609,
    ),  # eV->kcal/mol, eV/A -> kcal/mol/A
    "Hartree/Bohr -> kcal/mol/A": lambda energy, forces: (
        energy * 627.509,
        forces * 627.509 / 0.529177,
    ),  # Hartree -> kcal/mol, Hartree/Bohr -> kcal/mol/A
    "Hartree/A -> kcal/mol/A": lambda energy, forces: (
        energy * 627.509,
        forces * 627.509,
    ),  # Hartree -> kcal/mol, Hartree/A -> kcal/mol/A
}


class External:
    """This is an adapter to use TorchMD-Net models in TorchMD.

    Parameters
    ----------
    netfile : str or torch.nn.Module
        Path to the checkpoint file of the model or the model itself.
    embeddings : torch.Tensor
        Embeddings of the atoms in the system.
    device : str, optional
        Device on which the model should be run. Default: "cpu"
    output_transform : str or callable, optional
        Transform to apply to the energy and forces.
        If a string is given, it should be a key in the `transforms` dict.
        If a callable is given, it should take two arguments (energy and forces) and return two tensors of the same shape.
        Default: None
    use_cuda_graph : bool, optional
        Whether to use CUDA graphs to speed up the calculation. Default: False
    cuda_graph_warmup_steps : int, optional
        Number of steps to run as warmup before recording the CUDA graph. Default: 12
    dtype : torch.dtype or str, optional
        Cast the input to this dtype if defined. If passed as a string it should be a valid torch dtype. Default: torch.float32
    kwargs : dict, optional
        Extra arguments to pass to the model when loading it.
    """

    def __init__(
        self,
        netfile,
        embeddings,
        device="cpu",
        output_transform=None,
        use_cuda_graph=False,
        cuda_graph_warmup_steps=12,
        dtype=torch.float32,
        **kwargs,
    ):
        if isinstance(netfile, str):
            extra_args = kwargs
            if use_cuda_graph:
                warnings.warn(
                    "CUDA graphs are enabled, setting static_shapes=True and check_errors=False"
                )
                extra_args["static_shapes"] = True
                extra_args["check_errors"] = False
            self.model = load_model(
                netfile,
                device=device,
                derivative=True,
                **extra_args,
            )
        elif isinstance(netfile, torch.nn.Module):
            if kwargs:
                warnings.warn(
                    "Warning: extra arguments are being ignored when passing a torch.nn.Module"
                )
            self.model = netfile
        else:
            raise ValueError(
                f"Expected a path to a checkpoint file or a torch.nn.Module, got {type(netfile)}"
            )
        self.device = device
        self.n_atoms = embeddings.size(1)
        self.embeddings = embeddings.reshape(-1).to(device)
        self.batch = torch.arange(embeddings.size(0), device=device).repeat_interleave(
            embeddings.size(1)
        )
        self.model.eval()

        if not output_transform:
            self.output_transformer = lambda energy, forces: (
                energy,
                forces,
            )  # identity
        elif output_transform in tranforms.keys():
            self.output_transformer = tranforms[output_transform]
        else:
            self.output_transformer = eval(output_transform)
        if not torch.cuda.is_available() and use_cuda_graph:
            raise ValueError("CUDA graphs are only available if CUDA is")
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.cuda_graph = None
        self.energy = None
        self.forces = None
        self.box = None
        self.pos = None
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError:
                raise ValueError(f"Unknown torch dtype {dtype}")
        self.dtype = dtype

    def _init_cuda_graph(self):
        stream = torch.cuda.Stream()
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            for _ in range(self.cuda_graph_warmup_steps):
                self.energy, self.forces = self.model(
                    self.embeddings, self.pos, self.batch, self.box
                )
            with torch.cuda.graph(self.cuda_graph):
                self.energy, self.forces = self.model(
                    self.embeddings, self.pos, self.batch, self.box
                )

    def calculate(self, pos, box=None):
        """Calculate the energy and forces of the system.

        Parameters
        ----------
        pos : torch.Tensor
            Positions of the atoms in the system.
        box : torch.Tensor, optional
            Box vectors of the system. Default: None

        Returns
        -------
        energy : torch.Tensor
            Energy of the system.
        forces : torch.Tensor
            Forces on the atoms in the system.
        """
        pos = pos.to(self.device).to(self.dtype).reshape(-1, 3)
        if box is not None:
            box = box.to(self.device).to(self.dtype)
        if self.use_cuda_graph:
            if self.pos is None:
                self.pos = (
                    pos.clone()
                    .to(self.device)
                    .detach()
                    .requires_grad_(pos.requires_grad)
                )
            if self.box is None and box is not None:
                self.box = box.clone().to(self.device).to(self.dtype).detach()
            if self.cuda_graph is None:
                self._init_cuda_graph()
            assert (
                self.cuda_graph is not None
            ), "CUDA graph is not initialized. This should not had happened."
            with torch.no_grad():
                self.pos.copy_(pos)
                if box is not None:
                    self.box.copy_(box)
                self.cuda_graph.replay()
        else:
            self.energy, self.forces = self.model(self.embeddings, pos, self.batch, box)
        assert self.forces is not None, "The model is not returning forces"
        assert self.energy is not None, "The model is not returning energy"
        return self.output_transformer(
            self.energy.clone().detach(),
            self.forces.clone().reshape(-1, self.n_atoms, 3).detach(),
        )
