# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Checkpoint Nested Metadata
# sibling    : nested None and integer metadata returned alongside a Tensor from a HOP subgraph

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: nested None and integer metadata returned from checkpoint.
def region(x):
    return {"value": x.sin(), "metadata": (None, 3)}


def model(x):
    result = checkpoint(region, x, use_reentrant=False)
    valid = result["metadata"][0] is None and result["metadata"][1] == 3
    return result["value"] + (1.0 if valid else -1.0)


x = torch.tensor([-1.5, 0.0, 2.25])
eager = model(x)
compiled_model = torch.compile(model, backend="eager", fullgraph=True)
compiled = compiled_model(x)
torch.testing.assert_close(compiled, eager)
