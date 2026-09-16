# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Pytree symbolic capture
# sibling    : nested dict, list, and tuple pytrees passed through wrap while shape symbols are captured

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: nested pytree inputs with captured symbolic dimensions.
def fn(tree):
    rows = tree["sizes"][0].shape[0]
    cols = tree["sizes"][1].shape[0]

    def body(payload):
        left, metadata = payload
        right = metadata["right"]
        return left[:rows, :cols] + right[:rows, :cols]

    return wrap(body, tree["payload"])


def make_input(rows, cols):
    left = torch.arange(120, dtype=torch.float32).reshape(10, 12)
    right = torch.full((10, 12), 2.0)
    return {
        "sizes": [torch.ones(rows), torch.ones(cols)],
        "payload": (left, {"right": right}),
    }


samples = [make_input(4, 5), make_input(7, 9)]
expected = [fn(tree) for tree in samples]

all_results = []
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    results = [compiled_fn(tree) for tree in samples]
    for actual, eager in zip(results, expected):
        torch.testing.assert_close(actual, eager)
    all_results.append(results)

for run in all_results[1:]:
    for actual, first in zip(run, all_results[0]):
        torch.testing.assert_close(actual, first)
