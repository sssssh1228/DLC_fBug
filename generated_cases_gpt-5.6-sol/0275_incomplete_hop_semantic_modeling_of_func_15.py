# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Checkpoint masked-select dynamic output
# sibling    : torch.masked_select dynamic output combined with a symbolic input size inside a checkpoint HOP

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: masked_select dynamic output and symbolic size arithmetic in a HOP subgraph.
torch._dynamo.config.capture_dynamic_output_shape_ops = True


def body(x):
    selected = torch.masked_select(x, x > 0)
    return selected * x.shape[0]


def fn(x):
    return checkpoint(body, x, use_reentrant=False)


compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
inputs = (
    torch.tensor([-2.0, 1.0, 3.0]),
    torch.tensor([4.0, -1.0, 0.0, 2.0, -3.0]),
)
for value in inputs:
    eager = fn(value.clone())
    compiled = compiled_fn(value.clone())
    assert compiled.shape == eager.shape
    torch.testing.assert_close(compiled, eager)
