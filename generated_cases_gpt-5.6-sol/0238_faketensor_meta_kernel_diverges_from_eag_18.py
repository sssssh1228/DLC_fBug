# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Where scalar dtype promotion
# sibling    : torch.where Tensor-Scalar overload dtype promotion

import torch

# Sibling under test: wrapped Python-scalar dtype promotion in torch.where.
def fn(tensor):
    mask = tensor.remainder(2).eq(0)
    selected = torch.where(mask, tensor, 0.5)
    return selected, selected.sum()


x = torch.tensor([[-3, -2, -1], [0, 1, 2]], dtype=torch.int32)
eager_values, eager_sum = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual_values, actual_sum = compiled_fn(x)

assert actual_values.dtype == eager_values.dtype
assert actual_values.device == eager_values.device
assert actual_values.stride() == eager_values.stride()
assert actual_sum.dtype == eager_sum.dtype
assert actual_sum.device == eager_sum.device
torch.testing.assert_close(actual_values, eager_values)
torch.testing.assert_close(actual_sum, eager_sum)
