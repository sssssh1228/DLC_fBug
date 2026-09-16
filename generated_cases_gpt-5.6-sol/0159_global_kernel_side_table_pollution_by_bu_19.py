# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Torch reduction builtin contamination
# sibling    : torch.sum builtin function stored in kernel_side_table

# Sibling construct: torch.sum builtin function stored in kernel_side_table
import torch
from torch._inductor.codecache import kernel_side_table

kernel_side_table.add_kernel(torch.sum)

def fn(x):
    if x.numel() > 0:
        return torch.square(x).mean()
    return x.sum()

x = torch.randn(3, 5)
expected = fn(x)
compiled = torch.compile(fn, backend="eager")
actual = compiled(x)
torch.testing.assert_close(actual, expected)
