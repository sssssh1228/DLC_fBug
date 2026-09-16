# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Diagonal view stride and offset
# sibling    : Tensor.diagonal metadata for a transposed view

import torch

# Sibling under test: Tensor.diagonal metadata for a transposed view.
def fn(x):
    result = x.transpose(0, 1).diagonal(offset=1, dim1=0, dim2=1)
    return result, result.stride(), result.storage_offset()

x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())

torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1] == eager[1], (compiled[1], eager[1])
assert compiled[2] == eager[2], (compiled[2], eager[2])
assert compiled[0].dtype == eager[0].dtype
assert compiled[0].device == eager[0].device
