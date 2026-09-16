# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Math builtin contamination
# sibling    : math.sin builtin function stored in kernel_side_table

# Sibling construct: math.sin builtin function stored in kernel_side_table
import math
import torch
from torch._inductor.codecache import kernel_side_table

kernel_side_table.add_kernel(math.sin)

def fn(x):
    return torch.cos(x) + 1

x = torch.randn(8)
expected = fn(x)
compiled = torch.compile(fn, backend="eager")
actual = compiled(x)
torch.testing.assert_close(actual, expected)
