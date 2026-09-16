# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Enumerate With Negative Start
# sibling    : enumerate() with an explicit nonzero start argument

import torch

# Sibling construct: enumerate(iterable, start) with a negative start index.
def fn(x, y, z):
    values = (x, y, z)
    adjusted = [value + index for index, value in enumerate(values, start=-2)]
    return torch.stack(adjusted)


x = torch.tensor([2.0, 3.0])
y = torch.tensor([4.0, 5.0])
z = torch.tensor([6.0, 7.0])
eager = fn(x, y, z)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x, y, z)
torch.testing.assert_close(compiled, eager)
