# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Scalar extraction keeps container state
# sibling    : Tensor.item data-dependent scalar graph break

import torch

# Sibling construct: Tensor.item data-dependent scalar extraction.
def fn(x):
    scalar = x.sum().item()
    values = [scalar]
    values.append(values[0] + 4.0)
    return x * values[-1], tuple(values)

x = torch.tensor([1.0, 2.0, 3.0])
eager_tensor, eager_values = fn(x.clone())
torch._dynamo.reset()
compiled_tensor, compiled_values = torch.compile(fn, backend="eager")(x.clone())
assert torch.equal(compiled_tensor, eager_tensor), (compiled_tensor, eager_tensor)
assert compiled_values == eager_values, (compiled_values, eager_values)
