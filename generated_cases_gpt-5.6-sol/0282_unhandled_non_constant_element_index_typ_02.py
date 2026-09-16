# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic list slice bound
# sibling    : list slicing with a SymInt stop bound

import torch

# Sibling under test: list slicing with a SymInt stop bound.
def fn(x):
    values = [1, 2, 4, 8, 16, 32]
    return sum(values[1:x.shape[0]])

compiled = torch.compile(fn, backend="eager", dynamic=True)
for size in (3, 5):
    x = torch.ones(size)
    eager = fn(x)
    actual = compiled(x)
    assert actual == eager
    assert type(actual) is type(eager)
