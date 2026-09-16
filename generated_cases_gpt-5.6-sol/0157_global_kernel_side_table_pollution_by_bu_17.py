# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Operator builtin contamination
# sibling    : operator.add builtin function stored in kernel_side_table

# Sibling construct: operator.add builtin function stored in kernel_side_table
import operator
import torch
from torch._inductor.codecache import kernel_side_table

kernel_side_table.add_kernel(operator.add)

def fn(x, y):
    product = x * y
    return product.relu()

x = torch.randn(4, 4)
y = torch.randn(4, 4)
expected = fn(x, y)
compiled = torch.compile(fn, backend="eager")
actual = compiled(x, y)
torch.testing.assert_close(actual, expected)
