# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Map iterator protocol
# sibling    : map iterator

import torch

# Sibling under test: map iterator produced by map().
def fn(x):
    iterator = map(lambda value: value * value, (2, 3, 4))
    first = next(iter(iterator))
    second = next(iterator)
    return x + first - second

input_eager = torch.tensor([10.0, 20.0])
input_compiled = input_eager.clone()
expected = fn(input_eager)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(input_compiled)
torch.testing.assert_close(actual, expected)
