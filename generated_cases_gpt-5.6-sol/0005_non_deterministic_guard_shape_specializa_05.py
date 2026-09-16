# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Tensor Alias Guard Stability
# sibling    : tensor identity and aliasing guards around mutation

import torch
import torch._dynamo
from torch._dynamo.utils import counters

# Sibling under test: tensor identity and aliasing guards around mutation.
def fn(a, b):
    a.add_(1)
    return a + b


def make_args(alias):
    base = torch.arange(6, dtype=torch.float32)
    if alias:
        return base, base
    return base, base.clone()


def eager_results(pattern):
    results = []
    for alias in pattern:
        a, b = make_args(alias)
        results.append(fn(a, b))
    return results


def profile():
    return tuple(
        (name, tuple(sorted(counters.get(name, {}).items())))
        for name in ("frames", "stats", "graph_break")
    )


pattern = (False, True, False, True, True, False)
expected = eager_results(pattern)
profiles = []
for _ in range(3):
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    actual = []
    for alias in pattern:
        a, b = make_args(alias)
        actual.append(compiled(a, b))
    for eager_result, compiled_result in zip(expected, actual):
        torch.testing.assert_close(compiled_result, eager_result)
    profiles.append(profile())

assert all(item == profiles[0] for item in profiles[1:]), profiles
