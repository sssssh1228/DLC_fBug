# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Broadcast Relationship Stability
# sibling    : broadcast dimensions alternating between singleton and equal-size guards

import torch
from torch._dynamo.utils import counters

# Sibling under test: broadcast dimensions alternating between singleton and equal-size guards.
def fn(x, y):
    combined = x + y
    return torch.where(combined >= 0, combined, combined * 0.25).sum(dim=-1)

inputs = [
    (
        torch.arange(6.0).reshape(2, 3, 1) - 2,
        torch.arange(4.0).reshape(1, 1, 4) - 1,
    ),
    (
        torch.arange(10.0).reshape(5, 2, 1) - 4,
        torch.arange(15.0).reshape(5, 1, 3) - 5,
    ),
    (
        torch.arange(12.0).reshape(3, 4, 1) - 3,
        torch.arange(2.0).reshape(1, 1, 2) - 1,
    ),
    (
        torch.arange(8.0).reshape(2, 4, 1) - 2,
        torch.arange(10.0).reshape(2, 1, 5) - 3,
    ),
]
expected = [fn(x, y) for x, y in inputs]

def run_trial():
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    outputs = [compiled(x, y) for x, y in inputs]
    fingerprint = (
        tuple(sorted(counters["frames"].items())),
        tuple(sorted(counters["stats"].items())),
    )
    return outputs, fingerprint

run_trial()
trials = [run_trial() for _ in range(3)]
for outputs, _ in trials:
    for actual, eager in zip(outputs, expected):
        torch.testing.assert_close(actual, eager)
assert all(fingerprint == trials[0][1] for _, fingerprint in trials[1:])
