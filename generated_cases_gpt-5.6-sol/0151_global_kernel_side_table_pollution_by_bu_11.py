# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Torch cosine builtin pollution
# sibling    : torch.cos builtin function

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: torch.cos builtin function.
def fn(x):
    return torch.cos(x) + x.square()

kernel_side_table.reset_table()
try:
    kernel_side_table.add_kernel(torch.cos)
    x = torch.linspace(-2.0, 2.0, steps=8)
    eager = fn(x.clone())
    compiled = torch.compile(fn, backend="eager")(x.clone())
    torch.testing.assert_close(compiled, eager)
finally:
    kernel_side_table.reset_table()
