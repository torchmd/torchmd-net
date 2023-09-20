import os.path as osp
import torch
import importlib.machinery
library = "torchmdnet_neighbors"
# Find the specification for the library
spec = importlib.machinery.PathFinder().find_spec(
    library, [osp.dirname(__file__)]
)
# Check if the specification is found and load the library
if spec is not None:
    torch.ops.load_library(spec.origin)
else:
    raise ImportError(f"Could not find module '{library}' in {osp.dirname(__file__)}")
get_neighbor_pairs_kernel = torch.ops.torchmdnet_neighbors.get_neighbor_pairs
if int(torch.__version__.split('.')[0]) >= 2:
    import torch._dynamo as dynamo
    dynamo.disallow_in_graph(get_neighbor_pairs_kernel)
