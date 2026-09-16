# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Symbolic value in tuple.count
# sibling    : tuple.count with symbolic equality comparisons

import torch

# Sibling construct: tuple.count with symbolic equality comparisons.
def fn(x):
    n = x.shape[0]
    values = (4, n, n + 1, n)
    return x.cos() + values.count(n)

compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
for size in (4, 7):
    x = torch.arange(size, dtype=torch.float32)
    eager = fn(x.clone())
    actual = compiled(x.clone())
    torch.testing.assert_close(actual, eager)
