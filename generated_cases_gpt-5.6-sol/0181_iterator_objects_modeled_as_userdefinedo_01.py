# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Tuple Iterator Identity and Advancement
# sibling    : tuple_iterator

import torch


def fn(x):
    # Sibling under test: tuple_iterator.__iter__/__next__
    iterator = iter((x, x + 1, x + 2))
    same_iterator = iter(iterator)
    first = next(same_iterator)
    second = next(iterator)
    return first + second


x = torch.tensor([1.0, 3.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
