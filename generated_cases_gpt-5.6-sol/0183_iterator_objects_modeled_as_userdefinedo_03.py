# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Dictionary Key Iterator
# sibling    : dict_keyiterator

import torch


def fn(x):
    # Sibling under test: dict_keyiterator.__iter__/__next__
    values = {"left": x + 1, "right": x * 3}
    keys = iter(values.keys())
    first_key = next(keys)
    second_key = next(iter(keys))
    return values[first_key] + values[second_key]


x = torch.tensor([1.0, 5.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
