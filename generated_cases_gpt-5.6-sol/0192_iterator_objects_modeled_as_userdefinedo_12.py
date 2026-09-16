# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Map Iterator Advancement
# sibling    : map iterator (__iter__ and __next__)

import torch

# Sibling construct: map iterator advanced through iter() and next().
def fn(x):
    mapped = map(lambda value: value * 2, (x, x + 1, x + 2))
    iterator = iter(mapped)
    first = next(iterator)
    second = next(iterator)
    return first - second

x = torch.tensor(4.0)
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)(x)
torch.testing.assert_close(compiled, eager)
