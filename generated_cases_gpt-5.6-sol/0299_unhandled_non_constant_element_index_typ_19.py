# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic list slice bound
# sibling    : list slicing with a symbolic integer stop bound

import torch

# Sibling under test: list slicing with a symbolic shape bound.
def fn(x):
    values = [x + offset for offset in range(5)]
    selected = values[:x.shape[0]]
    return torch.stack(selected).sum(dim=0)

compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
for x in (torch.randn(2, 3), torch.randn(4, 3)):
    eager = fn(x)
    compiled = compiled_fn(x)
    torch.testing.assert_close(compiled, eager)
