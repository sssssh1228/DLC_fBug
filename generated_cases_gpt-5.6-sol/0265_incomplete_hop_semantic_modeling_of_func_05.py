# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Symbolic arange in vmap
# sibling    : SymInt-backed torch.arange length derived inside a vmap subgraph

import torch

# Sibling construct: SymInt-backed arange construction inside a vmap subgraph.
def per_row(row):
    width = row.shape[0]
    offsets = torch.arange(width, device=row.device, dtype=row.dtype)
    return row + offsets


def target(x):
    return torch.vmap(per_row)(x)


compiled_target = torch.compile(target, backend="eager", dynamic=True)
for shape in ((3, 4), (2, 7)):
    x = torch.randn(shape)
    eager = target(x)
    compiled = compiled_target(x)
    torch.testing.assert_close(compiled, eager)
