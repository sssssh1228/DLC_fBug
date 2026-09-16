# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Modulo Reshape Guard Stability
# sibling    : symbolic modulo, floor-division, and reshape guards

import torch
from torch._dynamo.utils import counters

# Sibling under test: symbolic modulo, floor-division, and reshape guards.
def fn(x):
    even_rows = x.shape[0] - (x.shape[0] % 2)
    trimmed = x.narrow(0, 0, even_rows)
    grouped = trimmed.reshape(even_rows // 2, 2, x.shape[1])
    return grouped.mean(dim=1)

inputs = [
    torch.arange(15.0).reshape(5, 3),
    torch.arange(32.0).reshape(8, 4),
    torch.arange(14.0).reshape(7, 2),
    torch.arange(24.0).reshape(6, 4),
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
