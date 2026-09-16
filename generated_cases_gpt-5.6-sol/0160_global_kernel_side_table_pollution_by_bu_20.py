# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Builtins callable contamination
# sibling    : builtins.abs builtin function stored in kernel_side_table

# Sibling construct: builtins.abs builtin function stored in kernel_side_table
import builtins
import torch
from torch._inductor.codecache import kernel_side_table

kernel_side_table.add_kernel(builtins.abs)

def fn(items):
    head, tail = items[0], items[1:]
    return head - torch.cat(tail, dim=0)

items = [torch.randn(2), torch.randn(1), torch.randn(1)]
expected = fn(items)
compiled = torch.compile(fn, backend="eager")
actual = compiled(items)
torch.testing.assert_close(actual, expected)
