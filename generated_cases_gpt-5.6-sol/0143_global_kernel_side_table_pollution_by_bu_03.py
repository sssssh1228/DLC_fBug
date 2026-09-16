# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Polluted operator.mul side-table entry
# sibling    : operator.mul C builtin with tuple unpacking

import operator
import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling construct: operator.mul C builtin with tuple unpacking.
kernel_side_table.reset_table()
kernel_side_table.add_kernel(operator.mul)

try:
    def fn(inputs):
        left, right = inputs
        product = operator.mul(left, right)
        return product - right

    inputs = (
        torch.tensor([2.0, -3.0, 4.0]),
        torch.tensor([5.0, 0.5, -2.0]),
    )
    expected = fn(inputs)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    actual = compiled_fn(inputs)
    torch.testing.assert_close(actual, expected)
finally:
    kernel_side_table.reset_table()
