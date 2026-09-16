# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Structured constant checkpoint return
# sibling    : Nested string, integer, and None returns as siblings of a scalar non-Tensor return

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: nested immutable constants returned beside a Tensor from a HOP subgraph.
def body(x):
    return {
        "value": x.square() + 1.0,
        "metadata": ("square", 2),
        "optional": None,
    }


def fn(x):
    result = checkpoint(body, x, use_reentrant=False)
    return (
        result["value"],
        result["metadata"][0],
        result["metadata"][1],
        result["optional"],
    )


x = torch.tensor([-3.0, -0.5, 2.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1:] == eager[1:]
