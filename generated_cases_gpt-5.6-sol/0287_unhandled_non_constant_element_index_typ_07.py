# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : List count with scalar tensor
# sibling    : list.count with a scalar-tensor element

import torch

# Sibling under test: list.count with a scalar-tensor element.
def fn(x, value):
    occurrences = [1, 2, 3, 2].count(value)
    return x + occurrences

x = torch.randn(3)
value = torch.tensor(2, dtype=torch.int64)
eager = fn(x, value)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x, value)
torch.testing.assert_close(compiled, eager)
