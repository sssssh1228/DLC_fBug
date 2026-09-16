# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Symbolic modulo branch stability
# sibling    : symbolic modulo and relational shape guards

import torch

# Sibling under test: symbolic modulo and relational shape guards.
def fn(x):
    rows = x.shape[0]
    if rows % 2 == 0:
        return x.sum(dim=1) * 2
    return x.sum(dim=1) - 3


def make_inputs():
    return [
        torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
        for n in (4, 5, 8, 5)
    ]


expected = [fn(x) for x in make_inputs()]
baseline = None
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    actual = [compiled_fn(x) for x in make_inputs()]
    for eager_result, compiled_result in zip(expected, actual):
        torch.testing.assert_close(compiled_result, eager_result)
    if baseline is not None:
        for previous, current in zip(baseline, actual):
            torch.testing.assert_close(current, previous)
    baseline = [result.clone() for result in actual]
