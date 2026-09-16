# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Range Iterator With Default
# sibling    : range_iterator and two-argument next() exhaustion semantics

import torch

# Sibling construct: range iterator with next(iterator, default) at exhaustion.
def fn(x):
    iterator = iter(range(2))
    first = next(iterator, -10)
    second = next(iterator, -10)
    exhausted = next(iterator, -10)
    return x * (first + second + exhausted)

x = torch.tensor(2.0)
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)(x)
torch.testing.assert_close(compiled, eager)
