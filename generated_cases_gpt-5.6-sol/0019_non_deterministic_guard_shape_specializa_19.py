# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Stride specialization stability
# sibling    : contiguity, stride, and view-layout guards

import torch

# Sibling under test: contiguity, stride, and view-layout guards.
def fn(x):
    if x.is_contiguous():
        return x.reshape(-1)[:4] + 1
    return x.contiguous().reshape(-1)[:4] - 1


def make_inputs():
    contiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    transposed = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    strided = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
    return [contiguous, transposed, strided, contiguous.clone()]


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
