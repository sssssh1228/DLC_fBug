# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Nested wrap closure lifting
# sibling    : Nested wrap bodies sharing captured symbolic dimensions

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: nested wrap bodies sharing captured symbolic dimensions.
def fn(x):
    rows = x.shape[0]
    cols = x.shape[1]

    def outer(t):
        row_bias = torch.arange(
            rows, device=t.device, dtype=t.dtype
        ).reshape(rows, 1)

        def inner(u):
            col_bias = torch.arange(
                cols, device=u.device, dtype=u.dtype
            ).reshape(1, cols)
            return u + row_bias + col_bias

        return wrap(inner, t)

    return wrap(outer, x)


shapes = [(3, 4), (5, 4), (2, 7)]
for _ in range(3):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for shape in shapes:
        x = torch.randn(shape)
        eager = fn(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, eager)
