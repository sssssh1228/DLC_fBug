# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Operator multiplication pollution
# sibling    : operator.mul builtin function

import operator
import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: operator.mul builtin function.
def fn(lhs, rhs):
    product = operator.mul(lhs, rhs)
    return torch.where(product > 0, product, -product)

kernel_side_table.reset_table()
try:
    kernel_side_table.add_kernel(operator.mul)
    lhs = torch.tensor([-2.0, 3.0, -4.0])
    rhs = torch.tensor([5.0, -6.0, -7.0])
    eager = fn(lhs.clone(), rhs.clone())
    compiled = torch.compile(fn, backend="eager")(lhs.clone(), rhs.clone())
    torch.testing.assert_close(compiled, eager)
finally:
    kernel_side_table.reset_table()
