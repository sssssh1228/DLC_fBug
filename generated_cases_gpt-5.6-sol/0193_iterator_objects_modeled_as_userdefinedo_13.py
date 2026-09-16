# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Filter Iterator Predicate
# sibling    : filter iterator (__next__ with predicate-driven skipping)

import torch

# Sibling construct: filter iterator using next() after predicate-driven skipping.
def fn(x):
    values = (0, 1, 2, 3, 4)
    filtered = filter(lambda value: value % 2 == 0, values)
    first = next(filtered)
    second = next(filtered)
    return x + first + second

x = torch.tensor(5.0)
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x)
torch.testing.assert_close(compiled, eager)
