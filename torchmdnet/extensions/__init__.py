# Place here any short extensions to torch that you want to use in your code.
# The extensions present in extensions.cpp will be automatically compiled in setup.py and loaded here.
# The extensions will be available under torch.ops.torchmdnet_extensions, but you can add wrappers here to make them more convenient to use.
import os.path as osp
import torch
import importlib.machinery

def _load_library(library):
    """ Load a dynamic library containing torch extensions from the given path.
    Args:
        library (str): The name of the library to load.
    """
    # Find the specification for the library
    spec = importlib.machinery.PathFinder().find_spec(
        library, [osp.dirname(__file__)]
    )
    # Check if the specification is found and load the library
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:
        raise ImportError(f"Could not find module '{library}' in {osp.dirname(__file__)}")

_load_library("torchmdnet_extensions")

@torch.jit.script
def is_current_stream_capturing():
    """
    Returns True if the current CUDA stream is capturing.
    Returns False if CUDA is not available or the current stream is not capturing.

    This utility is required because the builtin torch function that does this is not scriptable.
    """
    _is_current_stream_capturing = torch.ops.torchmdnet_extensions.is_current_stream_capturing
    return _is_current_stream_capturing()

get_neighbor_pairs_kernel = torch.ops.torchmdnet_extensions.get_neighbor_pairs

# For some unknown reason torch.compile is not able to compile this function
if int(torch.__version__.split('.')[0]) >= 2:
    import torch._dynamo as dynamo
    dynamo.disallow_in_graph(get_neighbor_pairs_kernel)
