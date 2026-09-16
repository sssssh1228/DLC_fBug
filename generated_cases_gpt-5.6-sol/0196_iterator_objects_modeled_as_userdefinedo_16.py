# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Reverse iterator protocol
# sibling    : list_reverseiterator via reversed()

import torch

# Sibling under test: list_reverseiterator produced by reversed().
def fn(x):
    iterator = reversed([2, 4, 6])
    first = next(iter(iterator))
    second = next(iterator)
    return x + first * 10 + second

input_eager = torch.tensor([1.0, 3.0])
input_compiled = input_eager.clone()
expected = fn(input_eager)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(input_compiled)
torch.testing.assert_close(actual, expected)
