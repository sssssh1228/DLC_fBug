# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Relational Shape Branch Stability
# sibling    : shape-dependent Python comparison and branch guards

import torch
from torch._dynamo.utils import counters

# Sibling under test: shape-dependent Python comparison and branch guards.
def fn(x):
    if x.shape[0] <= x.shape[1]:
        return x.sin().sum(dim=0)
    return x.cos().sum(dim=1)

inputs = [
    torch.arange(8.0).reshape(2, 4),
    torch.arange(15.0).reshape(5, 3),
    torch.arange(9.0).reshape(3, 3),
    torch.arange(12.0).reshape(6, 2),
]
expected = [fn(x) for x in inputs]

def run_trial():
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    outputs = [compiled(x) for x in inputs]
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
