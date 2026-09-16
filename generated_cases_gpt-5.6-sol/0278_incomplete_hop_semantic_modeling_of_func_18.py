# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Autograd Non-Differentiable Output
# sibling    : FunctionCtx.mark_non_differentiable used by setup_context with multiple outputs

import torch

# Sibling construct: FunctionCtx.mark_non_differentiable in setup_context.
class SquareWithSummary(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return x.square(), x.sum()

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        squared, summary = output
        ctx.save_for_backward(x)
        ctx.mark_non_differentiable(summary)

    @staticmethod
    def backward(ctx, grad_squared, grad_summary):
        (x,) = ctx.saved_tensors
        return 2.0 * x * grad_squared


def model(x):
    return SquareWithSummary.apply(x)


def run(fn):
    x = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
    squared, summary = fn(x)
    squared.sum().backward()
    return squared.detach(), summary.detach(), x.grad.detach()


eager_squared, eager_summary, eager_grad = run(model)
compiled_model = torch.compile(model, backend="eager", fullgraph=True)
compiled_squared, compiled_summary, compiled_grad = run(compiled_model)
torch.testing.assert_close(compiled_squared, eager_squared)
torch.testing.assert_close(compiled_summary, eager_summary)
torch.testing.assert_close(compiled_grad, eager_grad)
