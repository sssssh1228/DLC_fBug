# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Overlapping as_strided metadata
# sibling    : torch.as_strided with a nonzero storage offset

import torch

# Sibling under test: torch.as_strided with overlapping strides and a storage offset.
def fn(x):
    return torch.as_strided(x, size=(3, 4), stride=(2, 1), storage_offset=3)


def assert_same(actual, expected):
    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


x = torch.arange(20, dtype=torch.float64)
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled(x)
assert_same(actual, eager)
