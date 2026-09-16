# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Stale unary cosine builtin
# sibling    : torch.cos unary builtin function

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: torch.cos unary builtin function.
def fn(x):
    return torch.cos(x) + 1.0


def main():
    kernel_side_table.reset_table()
    try:
        kernel_side_table.add_kernel(torch.cos)
        x = torch.linspace(-2.0, 2.0, steps=9)
        eager = fn(x)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        actual = compiled(x)
        torch.testing.assert_close(actual, eager)
    finally:
        kernel_side_table.reset_table()


if __name__ == "__main__":
    main()
