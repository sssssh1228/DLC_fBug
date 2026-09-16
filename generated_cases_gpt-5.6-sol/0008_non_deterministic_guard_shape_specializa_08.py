# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Symbolic Slice Bound Determinism
# sibling    : floor-division-derived symbolic slice bounds

import torch
from torch._dynamo.utils import counters

# Sibling construct under test: floor-division-derived symbolic slice bounds.
def fn(x):
    midpoint = x.shape[0] // 2
    prefix = x[:midpoint]
    suffix = x[midpoint:]
    return prefix.sum() - suffix.sum()

inputs = [
    torch.arange(10.0).reshape(5, 2),
    torch.arange(16.0).reshape(8, 2),
    torch.arange(6.0).reshape(3, 2),
    torch.arange(14.0).reshape(7, 2),
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
