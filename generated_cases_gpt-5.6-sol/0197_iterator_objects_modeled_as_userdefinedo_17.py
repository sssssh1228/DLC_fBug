# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Range iterator protocol
# sibling    : range_iterator via iter(range(...))

import torch

# Sibling under test: range_iterator produced by iter(range(...)).
def fn(x):
    iterator = iter(range(3, 9, 2))
    first = next(iterator)
    second = next(iter(iterator))
    return x * first + second

input_eager = torch.tensor([2.0, 5.0])
input_compiled = input_eager.clone()
expected = fn(input_eager)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(input_compiled)
torch.testing.assert_close(actual, expected)
