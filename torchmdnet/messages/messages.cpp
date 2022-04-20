#include <torch/extension.h>

TORCH_LIBRARY(messages, m) {
    m.def("pass_messages(Tensor neighbors, Tensor messages, Tensor states) -> (Tensor messages)");
}