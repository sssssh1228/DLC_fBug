# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Broadcast Guard Stability
# sibling    : symbolic broadcasting compatibility guards

import torch
import torch._dynamo
from torch._dynamo.utils import counters

# Sibling under test: symbolic broadcasting compatibility guards.
def fn(x, y):
    return (x + y).sin()


def make_inputs():
    cases = []
    for rows, cols, mode in (
        (2, 3, "row"),
        (4, 3, "column"),
        (3, 5, "full"),
        (2, 5, "row"),
        (4, 3, "column"),
    ):
        x = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
        if mode == "row":
            y = torch.arange(cols, dtype=torch.float32).reshape(1, cols)
        elif mode == "column":
            y = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
        else:
            y = torch.ones(rows, cols, dtype=torch.float32)
        cases.append((x, y))
    return cases


def profile():
    return tuple(
        (name, tuple(sorted(counters.get(name, {}).items())))
        for name in ("frames", "stats", "graph_break")
    )


expected = [fn(x, y) for x, y in make_inputs()]
profiles = []
for _ in range(3):
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    actual = [compiled(x, y) for x, y in make_inputs()]
    for eager_result, compiled_result in zip(expected, actual):
        torch.testing.assert_close(compiled_result, eager_result)
    profiles.append(profile())

assert all(item == profiles[0] for item in profiles[1:]), profiles
