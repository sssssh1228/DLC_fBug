# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Math square-root builtin pollution
# sibling    : math.sqrt C builtin function

import math
import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: math.sqrt C builtin function.
def fn(x, scale):
    factor = math.sqrt(scale)
    return x * factor + 1.0

kernel_side_table.reset_table()
try:
    kernel_side_table.add_kernel(math.sqrt)
    x = torch.tensor([1.0, 4.0, 9.0])
    eager = fn(x.clone(), 16.0)
    compiled = torch.compile(fn, backend="eager")(x.clone(), 16.0)
    torch.testing.assert_close(compiled, eager)
finally:
    kernel_side_table.reset_table()
