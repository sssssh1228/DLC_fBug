# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Checkpoint Nested Constant Return
# sibling    : non-Tensor constants in a nested checkpoint return pytree

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: non-Tensor constants in a nested checkpoint return pytree.
def checkpoint_body(x):
    return {"tensor": x.sin(), "metadata": (1, True, None)}


def fn(x):
    return checkpoint(checkpoint_body, x, use_reentrant=False)


x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x.detach().clone().requires_grad_(True))

torch.testing.assert_close(actual["tensor"], eager["tensor"])
assert actual["metadata"] == eager["metadata"]
