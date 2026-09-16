# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Relational Shape Branch Determinism
# sibling    : symbolic relational guards in tensor-shape control flow

import torch
from torch._dynamo.utils import counters

# Sibling construct under test: symbolic relational guards in tensor-shape control flow.
def fn(x):
    if x.shape[0] <= 3:
        return x.sin() + 1.0
    return x.cos() - 1.0

inputs = [
    torch.arange(6.0).reshape(2, 3),
    torch.arange(15.0).reshape(5, 3),
    torch.arange(9.0).reshape(3, 3),
    torch.arange(12.0).reshape(4, 3),
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
