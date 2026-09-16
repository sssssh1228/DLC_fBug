# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : CPU scalar device propagation
# sibling    : torch.mul with a CPU zero-dimensional scalar and an accelerator tensor

import torch

# Sibling under test: CPU zero-dimensional scalar device propagation in torch.mul.
def fn(args):
    tensor, cpu_scalar = args
    return torch.mul(tensor, cpu_scalar)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
scalar = torch.tensor(2.5, dtype=torch.float32, device="cpu")

eager = fn((x, scalar))
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn((x, scalar))

assert actual.device == eager.device
assert actual.dtype == eager.dtype
assert actual.shape == eager.shape
assert actual.stride() == eager.stride()
torch.testing.assert_close(actual, eager)
