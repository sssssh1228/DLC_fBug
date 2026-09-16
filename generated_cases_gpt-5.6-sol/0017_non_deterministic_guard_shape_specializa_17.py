# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Nested sequence guard stability
# sibling    : list length, tuple unpacking, and element-value guards

import torch

# Sibling under test: list length, tuple unpacking, and element-value guards.
def fn(x, transforms):
    result = x
    for scale, bias in transforms:
        result = result * scale + bias
    return result


def make_calls():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return [
        (x.clone(), [(2.0, 1.0)]),
        (x.clone(), [(1.0, 0.0), (0.5, -2.0)]),
        (x.clone(), []),
        (x.clone(), [(2.0, 1.0)]),
    ]


expected = [fn(x, transforms) for x, transforms in make_calls()]
baseline = None
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    actual = [compiled_fn(x, transforms) for x, transforms in make_calls()]
    for eager_result, compiled_result in zip(expected, actual):
        torch.testing.assert_close(compiled_result, eager_result)
    if baseline is not None:
        for previous, current in zip(baseline, actual):
            torch.testing.assert_close(current, previous)
    baseline = [result.clone() for result in actual]
