# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic needle in list.index
# sibling    : list.index with a symbolic integer search value

import torch

# Sibling construct: list.index with a symbolic integer search value.
def fn(x):
    n = x.shape[0]
    values = [n + 2, 1, n, n + 1]
    return x.sum() + values.index(n)

compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
for size in (4, 7):
    x = torch.arange(size, dtype=torch.float32)
    eager = fn(x.clone())
    actual = compiled(x.clone())
    torch.testing.assert_close(actual, eager)
