# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic needle in list.remove
# sibling    : list.remove with a symbolic integer search value

import torch

# Sibling construct: list.remove with a symbolic integer search value.
def fn(x):
    n = x.shape[0]
    values = [n + 1, n, n + 2]
    values.remove(n)
    return x + values[0] + values[1]

compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
for size in (3, 6):
    x = torch.arange(size, dtype=torch.float32)
    eager = fn(x.clone())
    actual = compiled(x.clone())
    torch.testing.assert_close(actual, eager)
