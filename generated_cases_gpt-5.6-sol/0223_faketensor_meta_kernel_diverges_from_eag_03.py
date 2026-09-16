# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Noncontiguous complex real view
# sibling    : torch.view_as_real on a noncontiguous complex tensor

import torch

# Sibling under test: torch.view_as_real on a noncontiguous complex tensor.
def fn(x):
    return torch.view_as_real(x)


def assert_same(actual, expected):
    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


real = torch.arange(12, dtype=torch.float32).reshape(3, 4)
imag = real + 0.5
x = torch.complex(real, imag).transpose(0, 1)
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled(x)
assert_same(actual, eager)
