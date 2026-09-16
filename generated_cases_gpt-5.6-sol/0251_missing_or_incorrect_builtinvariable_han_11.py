# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Custom absolute value dispatch
# sibling    : abs() dispatch to __abs__

import torch

# Sibling under test: abs() dispatch to __abs__.
class Magnitude:
    def __init__(self, value):
        self.value = value

    def __abs__(self):
        return torch.sqrt(self.value * self.value)


def fn(x):
    return abs(Magnitude(x)) + 1.0


x = torch.tensor([-3.0, 0.5, 4.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
