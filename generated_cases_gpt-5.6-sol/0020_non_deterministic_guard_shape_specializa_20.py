# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Dynamic slice range stability
# sibling    : slice-bound, step, and resulting symbolic-length guards

import torch

# Sibling under test: slice-bound, step, and resulting symbolic-length guards.
def fn(x, start, stop, step):
    selected = x[start:stop:step]
    if selected.shape[0] >= 3:
        return selected.sum()
    return selected.prod()


def make_calls():
    x = torch.arange(1, 13, dtype=torch.float32)
    return [
        (x.clone(), 0, 10, 2),
        (x.clone(), 1, 8, 3),
        (x.clone(), 2, None, 2),
        (x.clone(), 3, 5, 1),
        (x.clone(), 0, 10, 2),
    ]


expected = [fn(x, start, stop, step) for x, start, stop, step in make_calls()]
baseline = None
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    actual = [
        compiled_fn(x, start, stop, step)
        for x, start, stop, step in make_calls()
    ]
    for eager_result, compiled_result in zip(expected, actual):
        torch.testing.assert_close(compiled_result, eager_result)
    if baseline is not None:
        for previous, current in zip(baseline, actual):
            torch.testing.assert_close(current, previous)
    baseline = [result.clone() for result in actual]
