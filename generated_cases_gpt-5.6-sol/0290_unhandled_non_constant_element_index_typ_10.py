# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : List slice deletion with symbolic bounds
# sibling    : list slice deletion using SymInt bounds

import torch

# Sibling under test: list slice deletion using SymInt bounds.
def fn(x):
    values = [x + 1, x + 2, x + 3, x + 4]
    del values[x.shape[0] - 3:x.shape[0] - 1]
    return values[0] + values[-1]

x = torch.randn(4)
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
compiled = compiled_fn(x)
torch.testing.assert_close(compiled, eager)
