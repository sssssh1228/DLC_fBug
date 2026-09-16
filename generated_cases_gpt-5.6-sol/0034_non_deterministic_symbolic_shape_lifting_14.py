# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Nested wrap captures
# sibling    : nested wrap bodies capturing symbols from different lexical scopes

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: nested wraps capturing symbolic dimensions across lexical scopes.
def fn(x):
    outer_size = x.shape[0]

    def outer_body(value):
        inner_size = value.shape[1]

        def inner_body(tensor):
            return tensor * (outer_size + inner_size)

        return wrap(inner_body, value).relu()

    return wrap(outer_body, x)


torch.manual_seed(3)
inputs = [
    torch.randn(2, 4),
    torch.randn(5, 7),
    torch.randn(8, 3),
]
expected = [fn(x) for x in inputs]

for _ in range(4):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for x, eager_result in zip(inputs, expected):
        compiled_result = compiled(x.clone())
        torch.testing.assert_close(compiled_result, eager_result)
