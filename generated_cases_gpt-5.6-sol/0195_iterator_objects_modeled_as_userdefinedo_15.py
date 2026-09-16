# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Dictionary Key Iterator State
# sibling    : dict_keyiterator (__iter__ identity and ordered advancement)

import torch

# Sibling construct: dictionary key iterator preserving identity and iteration state.
def fn(x):
    mapping = {"left": x, "middle": x + 2, "right": x + 4}
    keys = iter(mapping.keys())
    first = next(keys)
    same_iterator = iter(keys) is keys
    second = next(keys)
    return mapping[first] + mapping[second], same_iterator

x = torch.tensor(1.0)
eager_value, eager_identity = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled_value, compiled_identity = compiled_fn(x)
torch.testing.assert_close(compiled_value, eager_value)
assert compiled_identity == eager_identity
