# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Scalar tensor list pop index
# sibling    : list.pop with a zero-dimensional integer tensor index

import torch

# Sibling under test: list.pop with a scalar-tensor index.
def fn(x, index):
    values = [x + 1, x + 2, x + 3]
    selected = values.pop(index)
    return selected + len(values)

compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
x = torch.randn(3, 3)
for index in (torch.tensor(0), torch.tensor(2)):
    eager = fn(x, index)
    compiled = compiled_fn(x, index)
    torch.testing.assert_close(compiled, eager)
