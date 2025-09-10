# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

# Place here any short extensions to torch that you want to use in your code.
# The extensions present in extensions.cpp will be automatically compiled in setup.py and loaded here.
# The extensions will be available under torch.ops.torchmdnet_extensions, but you can add wrappers here to make them more convenient to use.
# Place here too any meta registrations for your extensions if required.

import torch
from pathlib import Path
from . import torchmdnet_extensions, ops
