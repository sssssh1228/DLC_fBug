# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Stale reduction builtin
# sibling    : torch.sum reduction builtin with structured output

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: torch.sum reduction builtin with structured output.
def fn(x):
    row_totals = torch.sum(x, dim=1)
    return {"totals": row_totals, "normalized": x / row_totals.unsqueeze(1)}


def main():
    kernel_side_table.reset_table()
    try:
        kernel_side_table.add_kernel(torch.sum)
        x = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
        eager = fn(x)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        actual = compiled(x)
        torch.testing.assert_close(actual, eager)
    finally:
        kernel_side_table.reset_table()


if __name__ == "__main__":
    main()
