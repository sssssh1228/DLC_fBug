# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Symbolic expression ordering
# sibling    : Multiple positional symbolic expressions derived from size and numel

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: multiple positional symbolic expressions derived from size and numel.
def body(x, width, elements):
    flat = x.reshape(elements)
    return flat.reshape(-1, width).tanh()


def fn(x):
    width = x.size(1)
    elements = x.numel()
    symbolic_args = (width, elements)
    return wrap(body, x, *symbolic_args)


shapes = [(3, 4), (5, 4), (2, 7)]
for _ in range(3):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for shape in shapes:
        x = torch.randn(shape)
        eager = fn(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, eager)
