# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Checkpoint callable instance
# sibling    : Callable object with __call__ as a sibling of functools.partial HOP callables

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: callable object implementing __call__ inside a checkpoint HOP.
class AffineSine:
    def __init__(self, scale, bias):
        self.scale = scale
        self.bias = bias

    def __call__(self, x):
        return torch.sin(x * self.scale) + self.bias


operation = AffineSine(1.75, -0.25)


def fn(x):
    return checkpoint(operation, x, use_reentrant=False)


x = torch.linspace(-2.0, 2.0, 9)
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
