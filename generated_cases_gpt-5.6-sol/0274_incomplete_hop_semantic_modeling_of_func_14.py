# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Conditional branch boolean short circuit
# sibling    : Dictionary membership, equality, and short-circuit boolean operations in a torch.cond branch

import torch

# Sibling construct: non-Tensor membership and short-circuit bool operations in a cond subgraph.
OPTIONS = {"enabled": True, "operation": "offset", "offset": 1.25}


def true_branch(x):
    should_offset = (
        "enabled" in OPTIONS
        and OPTIONS["enabled"]
        and OPTIONS.get("operation") == "offset"
    )
    if should_offset:
        return x + OPTIONS["offset"]
    return x


def false_branch(x):
    should_negate = "negate" not in OPTIONS or not OPTIONS.get("negate", False)
    if should_negate:
        return -x
    return x


def fn(x):
    return torch.cond(x.sum() > 0, true_branch, false_branch, (x,))


compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
for value in (
    torch.tensor([1.0, 2.0, -0.5]),
    torch.tensor([-3.0, 0.25, 0.5]),
):
    eager = fn(value.clone())
    compiled = compiled_fn(value.clone())
    torch.testing.assert_close(compiled, eager)
