# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Dtype reinterpretation view
# sibling    : Tensor.view dtype reinterpretation followed by stepped slicing

import torch

# Sibling under test: Tensor.view dtype reinterpretation followed by stepped slicing.
def fn(x):
    reinterpreted = x.view(torch.int16)
    return reinterpreted[:, 1::2]


def assert_same(actual, expected):
    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled(x)
assert_same(actual, eager)
