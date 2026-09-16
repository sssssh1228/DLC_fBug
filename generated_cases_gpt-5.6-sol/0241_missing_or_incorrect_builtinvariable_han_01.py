# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Absolute Value Dunder Dispatch
# sibling    : abs() dispatch to a user-defined __abs__ method

import torch

# Sibling construct: abs() dispatch through __abs__.
class Magnitude:
    def __init__(self, value):
        self.value = value

    def __abs__(self):
        return self.value * 2


def fn(x):
    return abs(Magnitude(x))


x = torch.tensor([-3.0, 1.5, 4.0])
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x)
torch.testing.assert_close(compiled, eager)
