# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Tensor and CPU scalar device propagation
# sibling    : multiplication with a CPU zero-dimensional scalar tensor

import torch

# Sibling under test: multiplication with a CPU zero-dimensional scalar tensor.
def fn(x, scalar):
    return x * scalar


def assert_same(actual, expected):
    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
scalar = torch.tensor(2.5, dtype=torch.float32, device="cpu")
eager = fn(x, scalar)
compiled = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled(x, scalar)
assert_same(actual, eager)
