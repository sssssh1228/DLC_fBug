# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Complex view reinterpretation
# sibling    : torch.view_as_complex on a transposed real tensor

import torch

# Sibling under test: dtype and stride reinterpretation in torch.view_as_complex.
def fn(tensor):
    complex_view = torch.view_as_complex(tensor)
    return complex_view.conj_physical()


base = torch.arange(48, dtype=torch.float32).reshape(2, 3, 4, 2)
x = base.transpose(0, 1)
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x)

assert actual.device == eager.device
assert actual.dtype == eager.dtype
assert actual.shape == eager.shape
assert actual.stride() == eager.stride()
assert actual.storage_offset() == eager.storage_offset()
torch.testing.assert_close(actual, eager)
