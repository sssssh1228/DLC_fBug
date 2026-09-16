# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Scalar extraction graph break
# sibling    : Tensor.item data-dependent scalar extraction

import torch

# Sibling construct: Tensor.item triggers a scalar-extraction graph break.
old_capture = torch._dynamo.config.capture_scalar_outputs
torch._dynamo.config.capture_scalar_outputs = False


def fn(x):
    scale = x.sum().item()
    if scale > 0:
        return x * scale
    return x - scale


try:
    compiled = torch.compile(fn, backend="eager")
    for value in (
        torch.tensor([1.0, 2.0, -0.5]),
        torch.tensor([-3.0, 0.5, 0.25]),
    ):
        eager = fn(value.clone())
        actual = compiled(value.clone())
        torch.testing.assert_close(actual, eager)
finally:
    torch._dynamo.config.capture_scalar_outputs = old_capture
