# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Map Iterator Protocol
# sibling    : map

import torch


def fn(x):
    # Sibling under test: map.__iter__/__next__
    iterator = map(torch.neg, (x, x + 4))
    first = next(iter(iterator))
    second = next(iterator)
    return first * second


x = torch.tensor([2.0, -3.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
