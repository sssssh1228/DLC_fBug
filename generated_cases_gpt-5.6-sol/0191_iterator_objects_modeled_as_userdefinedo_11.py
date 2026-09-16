# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Reversed Iterator Identity and Advancement
# sibling    : reversed iterator (__iter__ identity and __next__)

import torch

# Sibling construct: reversed iterator with iter() identity and next().
def fn(x):
    iterator = reversed([x, x + 1, x + 2])
    same_iterator = iter(iterator) is iterator
    first = next(iterator)
    second = next(iterator)
    return first + second, same_iterator

x = torch.tensor(3.0)
eager_value, eager_identity = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled_value, compiled_identity = compiled_fn(x)
torch.testing.assert_close(compiled_value, eager_value)
assert compiled_identity == eager_identity
