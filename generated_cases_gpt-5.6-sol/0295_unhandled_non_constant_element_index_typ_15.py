# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic index in list.insert
# sibling    : list.insert with a symbolic integer position

import torch

# Sibling construct: list.insert with a symbolic integer position.
def fn(x):
    index = x.shape[0] - 4
    values = [x + 1, x + 2]
    values.insert(index, x - 1)
    return torch.stack(values)

compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
for size in (4, 6):
    x = torch.arange(size, dtype=torch.float32)
    eager = fn(x.clone())
    actual = compiled(x.clone())
    torch.testing.assert_close(actual, eager)
