# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic tuple subscript
# sibling    : tuple.__getitem__ with a SymInt-derived index

import torch

# Sibling under test: tuple.__getitem__ with a SymInt-derived index.
def fn(x):
    values = (11, 22, 33, 44)
    return x + values[x.shape[0] - 2]

compiled = torch.compile(fn, backend="eager", dynamic=True)
for size in (3, 4):
    x = torch.arange(size, dtype=torch.float32)
    eager = fn(x)
    actual = compiled(x)
    torch.testing.assert_close(actual, eager)
