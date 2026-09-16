# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Inferred Reshape Range Determinism
# sibling    : symbolic multiplication and inferred-dimension reshape guards

import torch
from torch._dynamo.utils import counters

# Sibling construct under test: symbolic multiplication and inferred-dimension reshape guards.
def fn(x):
    rows = x.shape[0]
    flattened = x.reshape(rows, -1)
    return flattened.square().sum(dim=1)

inputs = [
    torch.arange(24.0).reshape(2, 3, 4),
    torch.arange(24.0).reshape(3, 2, 4),
    torch.arange(24.0).reshape(1, 6, 4),
    torch.arange(48.0).reshape(4, 3, 4),
]
expected = [fn(x) for x in inputs]
signatures = []
baseline = None

for _ in range(3):
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    actual = [compiled(x) for x in inputs]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    if baseline is None:
        baseline = [value.clone() for value in actual]
    else:
        for got, first in zip(actual, baseline):
            torch.testing.assert_close(got, first)
    signatures.append((
        counters["frames"].get("total", 0),
        counters["frames"].get("ok", 0),
        counters["stats"].get("calls_captured", 0),
        counters["stats"].get("unique_graphs", 0),
    ))

assert signatures.count(signatures[0]) == len(signatures), signatures
