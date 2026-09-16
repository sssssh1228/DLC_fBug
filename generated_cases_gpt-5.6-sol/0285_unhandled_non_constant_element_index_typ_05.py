# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Tensor-indexed list assignment
# sibling    : list.__setitem__ with a zero-dimensional integer tensor index

import torch

# Sibling under test: list.__setitem__ with a zero-dimensional integer tensor index.
def fn(index, replacement):
    values = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]
    values[index] = replacement
    return torch.stack(values)

compiled = torch.compile(fn, backend="eager")
for index in (torch.tensor(0), torch.tensor(-1)):
    replacement = torch.tensor(9)
    eager = fn(index, replacement)
    actual = compiled(index, replacement)
    torch.testing.assert_close(actual, eager)
