# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Symbolic Shape Branch Stability
# sibling    : symbolic relational shape guards

import torch
import torch._dynamo
from torch._dynamo.utils import counters

# Sibling under test: symbolic relational shape guards.
def fn(x):
    if x.shape[0] > 2:
        return x.cos() + 1
    return x.sin() - 1


def inputs():
    return [
        torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
        for n in (2, 4, 3, 1, 4, 2)
    ]


def profile():
    return tuple(
        (name, tuple(sorted(counters.get(name, {}).items())))
        for name in ("frames", "stats", "graph_break")
    )


expected = [fn(x) for x in inputs()]
profiles = []
for _ in range(3):
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    actual = [compiled(x) for x in inputs()]
    for eager_result, compiled_result in zip(expected, actual):
        torch.testing.assert_close(compiled_result, eager_result)
    profiles.append(profile())

assert all(item == profiles[0] for item in profiles[1:]), profiles
