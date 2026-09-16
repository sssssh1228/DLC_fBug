# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : List pop with symbolic index
# sibling    : list.pop with a SymInt index

import torch

# Sibling under test: list.pop with a SymInt index.
def fn(x):
    values = [x * 2, x * 3, x * 4]
    selected = values.pop(x.shape[0] - 2)
    return selected + values[0]

x = torch.randn(3)
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
compiled = compiled_fn(x)
torch.testing.assert_close(compiled, eager)
