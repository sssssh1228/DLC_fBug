# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Keyword symbolic size lifting
# sibling    : Keyword arguments carrying symbolic sizes into wrap

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: keyword arguments carrying symbolic sizes into wrap.
def body(x, *, rows):
    offsets = torch.arange(rows, device=x.device, dtype=x.dtype).unsqueeze(1)
    return x + offsets


def fn(x):
    return wrap(body, x, rows=x.shape[0])


shapes = [(3, 4), (5, 4), (2, 7)]
for _ in range(3):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for shape in shapes:
        x = torch.randn(shape)
        eager = fn(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, eager)
