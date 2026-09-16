# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Reversed Iterator Advancement
# sibling    : reversed

import torch


def fn(x):
    # Sibling under test: reversed.__iter__/__next__
    iterator = reversed([x, x + 2, x + 5])
    last = next(iterator)
    middle = next(iter(iterator))
    return last - middle


x = torch.tensor([3.0, 7.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
