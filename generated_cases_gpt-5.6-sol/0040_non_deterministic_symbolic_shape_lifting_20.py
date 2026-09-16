# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Derived symbolic expressions
# sibling    : closure factory capturing tuple-unpacked arithmetic expressions over symbolic sizes

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: tuple-unpacked derived symbolic expressions captured by a closure factory.
def make_body(bounds):
    row_size, col_size = bounds
    kept_rows = row_size - 1
    kept_cols = (col_size + 1) // 2

    def body(value):
        return value.narrow(0, 0, kept_rows).narrow(1, 0, kept_cols).square()

    return body


def fn(data, shape_sources):
    row_source, col_source = shape_sources
    bounds = (row_source.shape[0], col_source.shape[0])
    return wrap(make_body(bounds), data)


def make_inputs(rows, cols):
    data = torch.arange(196, dtype=torch.float32).reshape(14, 14)
    return data, (torch.ones(rows), torch.ones(cols))


samples = [make_inputs(5, 7), make_inputs(10, 12)]
expected = [fn(data, sources) for data, sources in samples]

all_results = []
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    results = [compiled_fn(data, sources) for data, sources in samples]
    for actual, eager in zip(results, expected):
        torch.testing.assert_close(actual, eager)
    all_results.append(results)

for run in all_results[1:]:
    for actual, first in zip(run, all_results[0]):
        torch.testing.assert_close(actual, first)
