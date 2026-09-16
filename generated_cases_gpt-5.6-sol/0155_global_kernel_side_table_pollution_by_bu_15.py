# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Bound list method pollution
# sibling    : list.append bound builtin method

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: list.append bound builtin method.
def fn(x):
    values = []
    values.append(x + 1)
    values.extend([x * 2])
    return values[0] + values[1]

kernel_side_table.reset_table()
try:
    stale_owner = []
    kernel_side_table.add_kernel(stale_owner.append)
    x = torch.tensor([2.0, -1.0, 4.0])
    eager = fn(x.clone())
    compiled = torch.compile(fn, backend="eager")(x.clone())
    torch.testing.assert_close(compiled, eager)
finally:
    kernel_side_table.reset_table()
