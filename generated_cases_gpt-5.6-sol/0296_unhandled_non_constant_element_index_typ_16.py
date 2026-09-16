# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic shape in list count
# sibling    : list.count with a symbolic integer search value

import torch

# Sibling under test: list.count with a symbolic shape value.
def fn(x):
    matches = [2, 4, 6, 8].count(x.shape[0])
    return x + matches

compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
for x in (torch.randn(4, 3), torch.randn(5, 3)):
    eager = fn(x)
    compiled = compiled_fn(x)
    torch.testing.assert_close(compiled, eager)
