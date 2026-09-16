# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Stale keyword add builtin
# sibling    : torch.add binary builtin with keyword arguments

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: torch.add binary builtin with keyword arguments.
def fn(x, y, scale=2.0):
    return torch.add(input=x, other=y, alpha=scale)


def main():
    kernel_side_table.reset_table()
    try:
        kernel_side_table.add_kernel(torch.add)
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        y = torch.full((2, 3), 0.5)
        eager = fn(x, y, scale=3.0)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        actual = compiled(x, y, scale=3.0)
        torch.testing.assert_close(actual, eager)
    finally:
        kernel_side_table.reset_table()


if __name__ == "__main__":
    main()
