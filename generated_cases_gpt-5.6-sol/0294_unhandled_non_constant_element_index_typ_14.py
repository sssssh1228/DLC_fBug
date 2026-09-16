# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Scalar tensor list repetition
# sibling    : list.__mul__ with a zero-dimensional integer tensor

import torch

# Sibling construct: list.__mul__ with a zero-dimensional integer tensor.
def fn(x, repeats):
    values = [x.sin(), x.cos()] * repeats
    return torch.stack(values).sum(dim=0)

compiled = torch.compile(fn, backend="eager", fullgraph=True)
x = torch.arange(5, dtype=torch.float32)
for count in (2, 3):
    repeats = torch.tensor(count, dtype=torch.int64)
    eager = fn(x.clone(), repeats.clone())
    actual = compiled(x.clone(), repeats.clone())
    torch.testing.assert_close(actual, eager)
