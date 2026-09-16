# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Symbolic Loop Trip Count Stability
# sibling    : Python range driven by a symbolic tensor dimension

import torch
from torch._dynamo.utils import counters

# Sibling under test: Python range driven by a symbolic tensor dimension.
def fn(x):
    accumulator = x.new_zeros(())
    for index in range(x.shape[0]):
        row_sum = x[index].sum()
        if index % 2 == 0:
            accumulator = accumulator + row_sum
        else:
            accumulator = accumulator - row_sum
    return accumulator

inputs = [
    torch.arange(6.0).reshape(2, 3),
    torch.arange(12.0).reshape(4, 3),
    torch.arange(9.0).reshape(3, 3),
    torch.arange(15.0).reshape(5, 3),
    torch.arange(6.0).reshape(2, 3),
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
