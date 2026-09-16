# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : CPU scalar tensor device propagation
# sibling    : Tensor.__mul__ with a CPU zero-dimensional scalar tensor

import torch

# Sibling under test: Tensor.__mul__ with a CPU zero-dimensional scalar tensor.
def fn(x, cpu_scalar):
    result = x * cpu_scalar
    return result, result.stride()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
cpu_scalar = torch.tensor(2.5, dtype=torch.float32, device="cpu")
eager = fn(x.clone(), cpu_scalar.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone(), cpu_scalar.clone())

torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1] == eager[1], (compiled[1], eager[1])
assert compiled[0].dtype == eager[0].dtype
assert compiled[0].device == eager[0].device
