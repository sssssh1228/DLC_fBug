# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Complex view-as-real metadata
# sibling    : torch.view_as_real on a noncontiguous complex tensor

import torch

# Sibling under test: torch.view_as_real on a noncontiguous complex tensor.
def fn(z):
    result = torch.view_as_real(z.transpose(0, 1))
    return result, result.shape, result.stride(), result.storage_offset()

real = torch.arange(24, dtype=torch.float32).reshape(4, 6)
imag = real.neg().add(3)
z = torch.complex(real, imag)
eager = fn(z.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(z.clone())

torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1] == eager[1], (compiled[1], eager[1])
assert compiled[2] == eager[2], (compiled[2], eager[2])
assert compiled[3] == eager[3], (compiled[3], eager[3])
assert compiled[0].dtype == eager[0].dtype
assert compiled[0].device == eager[0].device
