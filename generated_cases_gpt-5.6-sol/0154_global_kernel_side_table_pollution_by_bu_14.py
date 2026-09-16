# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Builtin length pollution
# sibling    : builtins.len builtin function

import builtins
import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: builtins.len builtin function.
def fn(x):
    pieces = (x + 1, x * 2, x - 3)
    return pieces[0] + pieces[len(pieces) - 1]

kernel_side_table.reset_table()
try:
    kernel_side_table.add_kernel(builtins.len)
    x = torch.arange(6, dtype=torch.float32)
    eager = fn(x.clone())
    compiled = torch.compile(fn, backend="eager")(x.clone())
    torch.testing.assert_close(compiled, eager)
finally:
    kernel_side_table.reset_table()
