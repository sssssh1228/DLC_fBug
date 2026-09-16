# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Chain iterator protocol
# sibling    : itertools.chain iterator

import itertools

import torch

# Sibling under test: itertools.chain iterator.
def fn(x):
    iterator = itertools.chain((1,), (3, 5), (7,))
    first = next(iter(iterator))
    second = next(iterator)
    third = next(iterator)
    return x + first + second * third

input_eager = torch.tensor([0.0, 4.0])
input_compiled = input_eager.clone()
expected = fn(input_eager)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(input_compiled)
torch.testing.assert_close(actual, expected)
