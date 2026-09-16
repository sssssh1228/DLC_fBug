# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Range Iterator Consumption
# sibling    : range_iterator

import torch


def fn(x):
    # Sibling under test: range_iterator.__iter__/__next__
    iterator = iter(range(2, 9, 3))
    first = next(iterator)
    second = next(iter(iterator))
    return x * first + second


x = torch.tensor([2.0, 4.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
