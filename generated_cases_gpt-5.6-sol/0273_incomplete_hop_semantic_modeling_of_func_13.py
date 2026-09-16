# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Autograd context with Python metadata
# sibling    : FunctionCtx saved tensor plus Python scalar metadata in setup_context

import torch

# Sibling construct: setup_context storing both a Tensor and non-Tensor metadata.
class ScaledCube(torch.autograd.Function):
    @staticmethod
    def forward(x, scale):
        return x.pow(3) * scale

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, scale = inputs
        ctx.save_for_backward(x)
        ctx.scale = scale

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output * 3.0 * x.square() * ctx.scale
        return grad_x, None


def fn(x):
    return ScaledCube.apply(x, 2.5).sin()


def run(callable_fn, source):
    x = source.clone().requires_grad_(True)
    output = callable_fn(x)
    output.sum().backward()
    return output.detach(), x.grad.detach()


source = torch.tensor([-1.5, -0.25, 0.75, 2.0])
eager_output, eager_grad = run(fn, source)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled_output, compiled_grad = run(compiled_fn, source)
torch.testing.assert_close(compiled_output, eager_output)
torch.testing.assert_close(compiled_grad, eager_grad)
