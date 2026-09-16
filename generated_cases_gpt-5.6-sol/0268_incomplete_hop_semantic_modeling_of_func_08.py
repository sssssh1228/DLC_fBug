# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Autograd Setup Context
# sibling    : torch.autograd.Function using separate setup_context and saved tensors

import torch

# Sibling construct: torch.autograd.Function using separate setup_context and saved tensors.
class CubicFunction(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return x * x * x

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 3 * x * x


def fn(x):
    return CubicFunction.apply(x).sum()


x_eager = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
eager = fn(x_eager)
eager.backward()
eager_grad = x_eager.grad.detach().clone()

x_compiled = x_eager.detach().clone().requires_grad_(True)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x_compiled)
actual.backward()

torch.testing.assert_close(actual, eager)
torch.testing.assert_close(x_compiled.grad, eager_grad)
