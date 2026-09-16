# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Polluted abs side-table entry
# sibling    : builtins.abs captured by a closure

import builtins
import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling construct: builtins.abs captured by a closure.
kernel_side_table.reset_table()
kernel_side_table.add_kernel(builtins.abs)

try:
    def make_fn(operation):
        def fn(x):
            magnitude = operation(x)
            return magnitude.square().sum()
        return fn

    fn = make_fn(builtins.abs)
    x = torch.tensor([-3.0, 0.0, 2.0])
    expected = fn(x)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    actual = compiled_fn(x)
    torch.testing.assert_close(actual, expected)
finally:
    kernel_side_table.reset_table()
