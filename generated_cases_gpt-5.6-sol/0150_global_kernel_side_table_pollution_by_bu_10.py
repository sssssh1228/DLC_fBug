# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : Global kernel_side_table pollution by builtin function
# title      : Stale ReLU builtin in module
# sibling    : torch.relu builtin invoked from nn.Module.forward

import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

# Sibling under test: torch.relu builtin invoked from nn.Module.forward.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.tensor([1.0, -1.0, 0.5]))

    def forward(self, x):
        residual = x
        return torch.relu(x + self.bias) + residual


def main():
    kernel_side_table.reset_table()
    try:
        kernel_side_table.add_kernel(torch.relu)
        model = Model().eval()
        x = torch.tensor([[-2.0, 0.5, 3.0], [1.0, 2.0, -4.0]])
        eager = model(x)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        actual = compiled(x)
        torch.testing.assert_close(actual, eager)
    finally:
        kernel_side_table.reset_table()


if __name__ == "__main__":
    main()
