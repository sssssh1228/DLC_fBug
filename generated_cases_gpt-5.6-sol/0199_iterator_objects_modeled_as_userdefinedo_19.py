# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Filter iterator protocol
# sibling    : filter iterator

import torch

# Sibling under test: filter iterator produced by filter().
def fn(x):
    iterator = filter(lambda value: value % 2 == 0, (1, 2, 3, 4, 5, 6))
    first = next(iterator)
    second = next(iter(iterator))
    return x * second + first

input_eager = torch.tensor([1.0, 2.0])
input_compiled = input_eager.clone()
expected = fn(input_eager)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(input_compiled)
torch.testing.assert_close(actual, expected)
