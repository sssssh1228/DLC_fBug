# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Symbolic Slice Bound Stability
# sibling    : shape-derived slice start, stop, and step guards

import torch
from torch._dynamo.utils import counters

# Sibling under test: shape-derived slice start, stop, and step guards.
def fn(x):
    start = x.shape[1] // 3
    stop = x.shape[1] - 1
    return x[:, start:stop:2].transpose(0, 1).contiguous()

inputs = [
    torch.arange(10.0).reshape(2, 5),
    torch.arange(24.0).reshape(3, 8),
    torch.arange(44.0).reshape(4, 11),
    torch.arange(14.0).reshape(2, 7),
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
