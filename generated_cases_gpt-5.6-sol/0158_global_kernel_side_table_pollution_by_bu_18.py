# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Tensor method descriptor contamination
# sibling    : torch.Tensor.add method descriptor stored in kernel_side_table

# Sibling construct: torch.Tensor.add method descriptor stored in kernel_side_table
import torch
from torch._inductor.codecache import kernel_side_table

kernel_side_table.add_kernel(torch.Tensor.add)

def fn(x):
    values = torch.stack((x, -x), dim=0)
    return values.sum(dim=0)

x = torch.randn(6)
expected = fn(x)
compiled = torch.compile(fn, backend="eager")
actual = compiled(x)
torch.testing.assert_close(actual, expected)
