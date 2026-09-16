# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Inlined helper closure lifting
# sibling    : Inlined local helper capturing a symbolic dimension

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: inlined local helper capturing a symbolic dimension.
def fn(x):
    width = x.shape[1]

    def helper(t, captured_width):
        bias = torch.arange(
            captured_width, device=t.device, dtype=t.dtype
        ).reshape(1, captured_width)
        return t.sin() + bias

    def body(t):
        return helper(t, width)

    return wrap(body, x)


shapes = [(3, 4), (5, 4), (2, 7)]
for _ in range(3):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for shape in shapes:
        x = torch.randn(shape)
        eager = fn(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, eager)
