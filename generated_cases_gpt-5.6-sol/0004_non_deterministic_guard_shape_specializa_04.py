# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Scalar Argument Specialization Stability
# sibling    : Python integer and boolean argument guards

import torch
import torch._dynamo
from torch._dynamo.utils import counters

# Sibling under test: Python integer and boolean argument guards.
def fn(x, dim, keepdim):
    return torch.sum(x, dim=dim, keepdim=keepdim)


def make_inputs():
    specs = (
        ((2, 3), 0, False),
        ((2, 3), 1, False),
        ((4, 3), 0, True),
        ((4, 5), 1, True),
        ((2, 5), 0, False),
        ((2, 3), 1, False),
    )
    return [
        (torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols), dim, keepdim)
        for (rows, cols), dim, keepdim in specs
    ]


def profile():
    return tuple(
        (name, tuple(sorted(counters.get(name, {}).items())))
        for name in ("frames", "stats", "graph_break")
    )


expected = [fn(x, dim, keepdim) for x, dim, keepdim in make_inputs()]
profiles = []
for _ in range(3):
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    actual = [compiled(x, dim, keepdim) for x, dim, keepdim in make_inputs()]
    for eager_result, compiled_result in zip(expected, actual):
        torch.testing.assert_close(compiled_result, eager_result)
    profiles.append(profile())

assert all(item == profiles[0] for item in profiles[1:]), profiles
