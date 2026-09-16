# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Partial Checkpoint Callable
# sibling    : functools.partial used as a checkpoint subgraph callable

import functools
import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: functools.partial used as a checkpoint subgraph callable.
def affine(scale, offset, x):
    return x * scale + offset


partially_bound_affine = functools.partial(affine, 2.5, -1.0)


def fn(x):
    return checkpoint(partially_bound_affine, x, use_reentrant=False)


x = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x.detach().clone().requires_grad_(True))

torch.testing.assert_close(actual, eager)
