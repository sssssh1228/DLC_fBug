# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : CPU scalar device propagation
# sibling    : torch.mul with a CPU zero-dimensional scalar and a device tensor

import torch

# Sibling under test: torch.mul CPU zero-dimensional scalar device propagation.
def fn(x, scale):
    return torch.mul(x, scale) + 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
scale = torch.tensor(2.5, dtype=torch.float32, device="cpu")

expected = fn(x, scale)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x, scale)

torch.testing.assert_close(actual, expected)
assert actual.device == expected.device
assert actual.dtype == expected.dtype
assert actual.shape == expected.shape
assert actual.stride() == expected.stride()
