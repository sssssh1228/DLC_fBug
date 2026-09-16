# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Preserve-format clone strides
# sibling    : Tensor.clone with torch.preserve_format on a noncontiguous dense view

import torch

# Sibling under test: Tensor.clone preserve-format stride computation.
def fn(x, *, preserve):
    if preserve:
        return x.clone(memory_format=torch.preserve_format)
    return x.clone(memory_format=torch.contiguous_format)


base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
x = base.transpose(0, 1)

expected = fn(x, preserve=True)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x, preserve=True)

torch.testing.assert_close(actual, expected)
assert actual.shape == expected.shape
assert actual.stride() == expected.stride()
assert actual.dtype == expected.dtype
assert actual.device == expected.device
assert actual.is_contiguous() == expected.is_contiguous()
