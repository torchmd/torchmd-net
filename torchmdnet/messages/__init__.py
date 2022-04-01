import os
import torch as pt
from torch.utils import cpp_extension

src_dir = os.path.dirname(__file__)
sources = ['messages.cpp', 'messages_cpu.cpp'] + (['messages_cuda.cu'] if pt.cuda.is_available() else [])
sources = [os.path.join(src_dir, name) for name in sources]

cpp_extension.load(name='messages', sources=sources, is_python_module=False)
pass_messages = pt.ops.messages.pass_messages