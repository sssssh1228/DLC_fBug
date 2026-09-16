# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Checkpoint Callable Object
# sibling    : stateful callable object passed as a higher-order-op subgraph function

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: stateful __call__ object used as a checkpoint function.
class Affine:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x, bias):
        return x * self.scale + bias


def model(x, bias):
    return checkpoint(Affine(2.5), x, bias, use_reentrant=False).cos()


x = torch.tensor([-2.0, 0.5, 3.0])
bias = torch.tensor([0.25, -1.0, 2.0])
eager = model(x, bias)
compiled_model = torch.compile(model, backend="eager", fullgraph=True)
compiled = compiled_model(x, bias)
torch.testing.assert_close(compiled, eager)
