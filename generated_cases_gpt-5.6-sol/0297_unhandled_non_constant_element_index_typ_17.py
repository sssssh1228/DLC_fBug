# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic shape in tuple index
# sibling    : tuple.index with a symbolic integer search value

import torch

# Sibling under test: tuple.index with a symbolic shape value.
def fn(x):
    position = (2, 4, 5, 7).index(x.shape[0])
    return x * (position + 1)

compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
for x in (torch.randn(4, 2), torch.randn(5, 2)):
    eager = fn(x)
    compiled = compiled_fn(x)
    torch.testing.assert_close(compiled, eager)
