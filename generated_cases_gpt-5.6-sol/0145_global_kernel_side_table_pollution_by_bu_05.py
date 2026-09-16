# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Polluted torch.minimum across branch recompilation
# sibling    : torch.minimum builtin selected by Python control flow

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling construct: torch.minimum builtin selected by Python control flow.
kernel_side_table.reset_table()
kernel_side_table.add_kernel(torch.minimum)

try:
    def fn(x, y, use_minimum):
        if use_minimum:
            return torch.minimum(x, y)
        return torch.maximum(x, y)

    x = torch.tensor([1.0, 5.0, -2.0])
    y = torch.tensor([3.0, 4.0, -7.0])
    expected_maximum = fn(x, y, False)
    expected_minimum = fn(x, y, True)

    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    actual_maximum = compiled_fn(x, y, False)
    actual_minimum = compiled_fn(x, y, True)

    torch.testing.assert_close(actual_maximum, expected_maximum)
    torch.testing.assert_close(actual_minimum, expected_minimum)
finally:
    kernel_side_table.reset_table()
