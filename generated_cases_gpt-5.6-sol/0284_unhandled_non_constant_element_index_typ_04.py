# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Scalar tensor tuple count
# sibling    : tuple.count with a zero-dimensional tensor search value

import torch

# Sibling under test: tuple.count with a zero-dimensional tensor search value.
def fn(needle):
    return (2, 4, 4, 8).count(needle)

compiled = torch.compile(fn, backend="eager")
for needle in (torch.tensor(4), torch.tensor(3)):
    eager = fn(needle)
    actual = compiled(needle)
    assert actual == eager
    assert type(actual) is type(eager)
