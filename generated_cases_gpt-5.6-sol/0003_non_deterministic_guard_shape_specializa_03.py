# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Symbolic Slice Bound Stability
# sibling    : symbolic floor-division and slice-bound guards

import torch
import torch._dynamo
from torch._dynamo.utils import counters

# Sibling under test: symbolic floor-division and slice-bound guards.
def fn(x):
    stop = (x.shape[1] + 1) // 2
    return x[:, 1:stop] * 2


def inputs():
    return [
        torch.arange(2 * cols, dtype=torch.float32).reshape(2, cols)
        for cols in (4, 5, 8, 3, 7, 4)
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
