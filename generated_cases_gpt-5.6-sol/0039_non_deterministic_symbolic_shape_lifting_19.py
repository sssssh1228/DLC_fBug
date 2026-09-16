# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Nested wrap symbolic bounds
# sibling    : nested wrap higher-order operations capturing different symbolic dimensions

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: nested wraps capturing independent symbolic row and column bounds.
def fn(data, row_source, col_source):
    rows = row_source.shape[0]
    cols = col_source.shape[0]

    def outer(value):
        cropped_rows = value[:rows]

        def inner(nested_value):
            return nested_value[:, :cols].tanh()

        return wrap(inner, cropped_rows)

    return wrap(outer, data)


def make_inputs(rows, cols):
    data = torch.linspace(-2.0, 2.0, 144).reshape(12, 12)
    return data, torch.ones(rows), torch.ones(cols)


samples = [make_inputs(3, 8), make_inputs(9, 5)]
expected = [fn(data, row_source, col_source) for data, row_source, col_source in samples]

all_results = []
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    results = [
        compiled_fn(data, row_source, col_source)
        for data, row_source, col_source in samples
    ]
    for actual, eager in zip(results, expected):
        torch.testing.assert_close(actual, eager)
    all_results.append(results)

for run in all_results[1:]:
    for actual, first in zip(run, all_results[0]):
        torch.testing.assert_close(actual, first)
