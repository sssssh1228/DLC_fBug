# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Scalar tensor list membership
# sibling    : list.__contains__ with a zero-dimensional tensor element

import torch

# Sibling under test: list.__contains__ with a zero-dimensional tensor element.
def fn(needle):
    return needle in [2, 4, 6]

compiled = torch.compile(fn, backend="eager")
for needle in (torch.tensor(4), torch.tensor(5)):
    eager = fn(needle)
    actual = compiled(needle)
    assert actual == eager
    assert type(actual) is type(eager)
