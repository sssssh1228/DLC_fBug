# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Non-differentiable autograd output
# sibling    : FunctionCtx.mark_non_differentiable with setup_context and saved tensors

import torch

# Sibling construct: FunctionCtx.mark_non_differentiable in setup_context.
class TaggedSin(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return x.sin(), x.argmax()

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        _, index = output
        ctx.save_for_backward(x)
        ctx.mark_non_differentiable(index)

    @staticmethod
    def backward(ctx, grad_value, grad_index):
        (x,) = ctx.saved_tensors
        return grad_value * x.cos()


def target(x):
    value, index = TaggedSin.apply(x)
    return value * 2.0, index


def run(fn, source):
    x = source.clone().requires_grad_(True)
    value, index = fn(x)
    value.sum().backward()
    return value.detach(), index.detach(), x.grad.detach()


source = torch.tensor([-0.75, 0.25, 1.5, -1.0])
eager_value, eager_index, eager_grad = run(target, source)
compiled_target = torch.compile(target, backend="eager")
compiled_value, compiled_index, compiled_grad = run(compiled_target, source)
torch.testing.assert_close(compiled_value, eager_value)
torch.testing.assert_close(compiled_index, eager_index)
torch.testing.assert_close(compiled_grad, eager_grad)
