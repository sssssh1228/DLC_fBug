# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Nested metadata checkpoint return
# sibling    : nested None, string, and integer constants returned beside a Tensor

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: nested immutable non-Tensor values returned from a checkpoint body.
def body(x):
    return {"value": x.cos(), "metadata": (None, "cosine", 3)}


def target(x):
    return checkpoint(body, x, use_reentrant=False)


x = torch.randn(6)
eager = target(x)
compiled = torch.compile(target, backend="eager")(x)
assert compiled["metadata"] == eager["metadata"]
torch.testing.assert_close(compiled["value"], eager["value"])
