# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Callable object in checkpoint
# sibling    : stateful __call__ object as a HOP subgraph callable

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: stateful __call__ object in a checkpoint subgraph.
class AffineSine:
    def __init__(self, scale, bias):
        self.scale = scale
        self.bias = bias

    def __call__(self, x):
        return torch.sin(x * self.scale) + self.bias


operation = AffineSine(1.75, -0.25)


def target(x):
    return checkpoint(operation, x, use_reentrant=False)


x = torch.linspace(-2.0, 2.0, 9)
eager = target(x)
compiled = torch.compile(target, backend="eager")(x)
torch.testing.assert_close(compiled, eager)
