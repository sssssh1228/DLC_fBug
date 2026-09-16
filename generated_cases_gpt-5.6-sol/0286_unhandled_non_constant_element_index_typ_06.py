# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Tuple index with symbolic shape
# sibling    : tuple.index with a SymInt search value

import torch

# Sibling under test: tuple.index with a SymInt search value.
def fn(x):
    position = (2, 4, 6).index(x.shape[0])
    return x + position

x = torch.randn(4)
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
compiled = compiled_fn(x)
torch.testing.assert_close(compiled, eager)
