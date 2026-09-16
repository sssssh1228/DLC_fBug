# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Inlined helper symbolic extent
# sibling    : an inlined helper and local function capturing a symbolic size in wrap

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: inlined helper function producing a captured symbolic extent.
def symbolic_extent(source):
    return source.shape[0]


def apply_crop(data, source):
    extent = symbolic_extent(source)

    def body(value):
        return torch.relu(value[:extent] - 3.0)

    return wrap(body, data)


def fn(data, source):
    return apply_crop(data, source) * 2.0


samples = [
    (torch.arange(12, dtype=torch.float32), torch.ones(4)),
    (torch.arange(15, dtype=torch.float32), torch.ones(9)),
]
expected = [fn(data, source) for data, source in samples]

all_results = []
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    results = [compiled_fn(data, source) for data, source in samples]
    for actual, eager in zip(results, expected):
        torch.testing.assert_close(actual, eager)
    all_results.append(results)

for run in all_results[1:]:
    for actual, first in zip(run, all_results[0]):
        torch.testing.assert_close(actual, first)
