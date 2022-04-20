#include <torch/extension.h>

using torch::kInt32;
using torch::logical_and;
using torch::Tensor;

static Tensor forward(const Tensor& neighbors, const Tensor& messages, const Tensor& states) {

    TORCH_CHECK(neighbors.dim() == 2, "Expected \"neighbors\" to have two dimensions");
    TORCH_CHECK(neighbors.size(0) == 2, "Expected the 2nd dimension size of \"neighbors\" to be 2");
    TORCH_CHECK(neighbors.scalar_type() == kInt32, "Expected \"neighbors\" to have data type of int32");
    TORCH_CHECK(neighbors.is_contiguous(), "Expected \"neighbors\" to be contiguous");

    TORCH_CHECK(messages.dim() == 2, "Expected \"messages\" to have two dimensions");
    TORCH_CHECK(messages.size(1) % 32 == 0, "Expected the 2nd dimension size of \"messages\" to be a multiple of 32");
    TORCH_CHECK(messages.size(1) <= 1024, "Expected the 2nd dimension size of \"messages\" to be less than 1024");
    TORCH_CHECK(messages.is_contiguous(), "Expected \"messages\" to be contiguous");

    TORCH_CHECK(states.dim() == 2, "Expected \"states\" to have two dimensions");
    TORCH_CHECK(states.size(1) == messages.size(1), "Expected the 2nd dimension size of \"messages\" and \"states\" to be the same");
    TORCH_CHECK(states.scalar_type() == messages.scalar_type(), "Expected the data type of \"messages\" and \"states\" to be the same");
    TORCH_CHECK(states.is_contiguous(), "Expected \"messages\" to be contiguous");

    const Tensor rows = neighbors[0];
    const Tensor columns = neighbors[1];

    const int num_features = messages.size(1);

    const Tensor mask = logical_and(rows > -1, columns > -1);
    const Tensor masked_rows = rows.masked_select(mask).to(torch::kLong);
    const Tensor masked_columns = columns.masked_select(mask).to(torch::kLong);
    const Tensor masked_messages = messages.masked_select(mask.unsqueeze(1)).reshape({-1, num_features});

    Tensor new_states = states.clone();
    new_states.index_add_(0, masked_rows, masked_messages);
    new_states.index_add_(0, masked_columns, masked_messages);

    return new_states;
}

TORCH_LIBRARY_IMPL(messages, CPU, m) {
    m.impl("pass_messages", &forward);
}