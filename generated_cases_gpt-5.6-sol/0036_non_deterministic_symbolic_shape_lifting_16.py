# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Keyword-only symbolic bounds
# sibling    : keyword-only arguments whose symbolic sizes are captured by a wrap body

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: keyword-only arguments captured as symbolic wrap bounds.
def fn(data, *, row_source, col_source):
    rows = row_source.shape[0]
    cols = col_source.shape[0]
    return wrap(lambda value: value[:rows, :cols].cos(), data)


def make_inputs(rows, cols):
    data = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    return data, torch.ones(rows), torch.ones(cols)


samples = [make_inputs(3, 4), make_inputs(6, 7)]
expected = [
    fn(data, row_source=row_source, col_source=col_source)
    for data, row_source, col_source in samples
]

all_results = []
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    results = [
        compiled_fn(data, row_source=row_source, col_source=col_source)
        for data, row_source, col_source in samples
    ]
    for actual, eager in zip(results, expected):
        torch.testing.assert_close(actual, eager)
    all_results.append(results)

for run in all_results[1:]:
    for actual, first in zip(run, all_results[0]):
        torch.testing.assert_close(actual, first)
