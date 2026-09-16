# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : As-strided storage metadata
# sibling    : Tensor.as_strided with an explicit storage offset

import torch

# Sibling under test: explicit stride and storage-offset modeling in Tensor.as_strided.
def fn(tensor):
    return tensor.as_strided(size=(3, 2), stride=(7, 3), storage_offset=2)


x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x)

assert actual.device == eager.device
assert actual.dtype == eager.dtype
assert actual.shape == eager.shape
assert actual.stride() == eager.stride()
assert actual.storage_offset() == eager.storage_offset()
torch.testing.assert_close(actual, eager)
