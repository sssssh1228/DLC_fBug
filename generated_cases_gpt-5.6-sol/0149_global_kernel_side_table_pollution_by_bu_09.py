# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Stale negation builtin in comprehension
# sibling    : torch.neg unary builtin inside a list comprehension

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: torch.neg unary builtin inside a list comprehension.
def fn(values):
    negated = [torch.neg(value) for value in values]
    return torch.stack(negated, dim=0)


def main():
    kernel_side_table.reset_table()
    try:
        kernel_side_table.add_kernel(torch.neg)
        values = [torch.tensor([1.0, -2.0]), torch.tensor([3.0, -4.0])]
        eager = fn(values)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        actual = compiled(values)
        torch.testing.assert_close(actual, eager)
    finally:
        kernel_side_table.reset_table()


if __name__ == "__main__":
    main()
