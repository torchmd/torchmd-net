#include <torch/extension.h>

using torch::kInt32;
using torch::logical_and;
using torch::Tensor;

static Tensor forward(const Tensor& neighbors, const Tensor& messages) {

    TORCH_CHECK(neighbors.dim() == 2, "Expected \"neighbors\" to have two dimensions");
    TORCH_CHECK(neighbors.size(0) == 2, "Expected the 2nd dimension size of \"neighbors\" to be 2");
    TORCH_CHECK(neighbors.scalar_type() == kInt32, "Expected \"neighbors\" to have data type of int32");
    TORCH_CHECK(neighbors.is_contiguous(), "Expected \"neighbors\" to be contiguous");

    TORCH_CHECK(messages.dim() == 2, "Expected \"messages\" to have two dimensions");
    TORCH_CHECK(messages.size(1) % 32 == 0, "Expected the 2nd dimension size of \"messages\" to be a multiple of 32");
    TORCH_CHECK(messages.size(1) <= 1024, "Expected the 2nd dimension size of \"messages\" to be less than 1024");
    TORCH_CHECK(messages.is_contiguous(), "Expected \"messages\" to be contiguous");

    const Tensor all_rows = neighbors[0];
    const Tensor all_columns = neighbors[1];

    const Tensor mask = logical_and(all_rows > -1, all_columns > -1);
    const Tensor rows = all_rows.masked_select(mask).to(torch::kLong);
    const Tensor columns = all_columns.masked_select(mask).to(torch::kLong);

    Tensor new_messages = messages.clone();
    new_messages.index_add_(0, rows, messages.index({columns}));
    new_messages.index_add_(0, columns, messages.index({rows}));

    return new_messages;
}

TORCH_LIBRARY_IMPL(messages, CPU, m) {
    m.impl("pass_messages", &forward);
}