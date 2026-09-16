# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Pytree wrap capture
# sibling    : nested dictionary and list pytree input with symbolic slicing

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: nested pytree wrap input with captured symbolic slice bounds.
def fn(x, y):
    width = x.shape[1]
    half = width // 2
    tree = {"left": x, "nested": [y]}

    def body(values):
        left = values["left"]
        right = values["nested"][0]
        return {
            "prefix": (left + right)[:, :half],
            "reduction": left.sum(dim=1) + width,
        }

    return wrap(body, tree)


torch.manual_seed(1)
inputs = [
    (torch.randn(3, 4), torch.randn(3, 4)),
    (torch.randn(3, 8), torch.randn(3, 8)),
    (torch.randn(3, 10), torch.randn(3, 10)),
]
expected = [fn(x, y) for x, y in inputs]

for _ in range(4):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for (x, y), eager_result in zip(inputs, expected):
        compiled_result = compiled(x.clone(), y.clone())
        torch.testing.assert_close(compiled_result, eager_result)
