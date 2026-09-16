# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Polluted torch.add side-table entry
# sibling    : torch.add builtin binary function used as a default argument

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling construct: torch.add builtin binary function used as a default argument.
kernel_side_table.reset_table()
kernel_side_table.add_kernel(torch.add)

try:
    def fn(x, y, operation=torch.add):
        return operation(x, y, alpha=2)

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([0.5, -1.0, 4.0])
    expected = fn(x, y)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    actual = compiled_fn(x, y)
    torch.testing.assert_close(actual, expected)
finally:
    kernel_side_table.reset_table()
