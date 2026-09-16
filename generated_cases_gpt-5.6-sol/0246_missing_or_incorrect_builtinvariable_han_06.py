# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Custom absolute-value dispatch
# sibling    : abs() dispatch to a user-defined __abs__ method

import torch

# Sibling under test: abs() dispatch through __abs__.
class Magnitude:
    def __init__(self, value):
        self.value = value

    def __abs__(self):
        return torch.sqrt(torch.sum(self.value * self.value))


def fn(x):
    return abs(Magnitude(x))


x = torch.tensor([3.0, 4.0])
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)(x)
assert torch.allclose(compiled, eager), (compiled, eager)
