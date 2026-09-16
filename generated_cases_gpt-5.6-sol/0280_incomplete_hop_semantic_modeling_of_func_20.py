# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Grad With Dynamic Masked Select
# sibling    : dynamic-output-shape masked_select and symbolic result size inside torch.func.grad

import torch

# Sibling construct: dynamic-shape masked_select inside a torch.func.grad body.
torch._dynamo.config.capture_dynamic_output_shape_ops = True


def loss(x):
    selected = torch.masked_select(x, x > 0)
    symbolic_count = selected.shape[0]
    return selected.square().sum() + symbolic_count * 0.0


def model(x):
    return torch.func.grad(loss)(x)


compiled_model = torch.compile(
    model,
    backend="eager",
    fullgraph=True,
    dynamic=True,
)
inputs = (
    torch.tensor([-2.0, 1.0, 3.0, -4.0]),
    torch.tensor([2.0, 1.0, -3.0, 4.0]),
)
for x in inputs:
    eager = model(x)
    compiled = compiled_model(x)
    torch.testing.assert_close(compiled, eager)
