# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Where scalar dtype promotion
# sibling    : torch.where dtype promotion with a wrapped Python scalar

import torch

# Sibling under test: torch.where wrapped-scalar dtype promotion.
def make_fn(limit, fallback):
    def fn(x):
        mask = x > limit
        return torch.where(mask, x, fallback)

    return fn


fn = make_fn(limit=0, fallback=-2.25)
x = torch.tensor([-3, -1, 0, 2, 7], dtype=torch.int16)

expected = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x)

torch.testing.assert_close(actual, expected)
assert actual.dtype == expected.dtype
assert actual.device == expected.device
assert actual.shape == expected.shape
assert actual.stride() == expected.stride()
