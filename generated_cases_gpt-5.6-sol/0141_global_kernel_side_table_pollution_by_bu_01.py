# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Polluted torch.cos side-table entry
# sibling    : torch.cos builtin unary function

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling construct: torch.cos builtin unary function.
kernel_side_table.reset_table()
kernel_side_table.add_kernel(torch.cos)

try:
    def fn(x):
        return torch.cos(x) + 1.0

    x = torch.tensor([-1.0, 0.0, 2.0])
    expected = fn(x)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    actual = compiled_fn(x)
    torch.testing.assert_close(actual, expected)
finally:
    kernel_side_table.reset_table()
