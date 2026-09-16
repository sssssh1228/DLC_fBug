# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Custom absolute-value dispatch
# sibling    : abs() dispatch to __abs__

import torch


class AbsoluteBox:
    def __init__(self, value):
        self.value = value

    def __abs__(self):
        return torch.where(self.value < 0, -self.value, self.value)


def fn(x):
    # Sibling under test: abs() dispatch to __abs__.
    return abs(AbsoluteBox(x)) + 1


x = torch.tensor([-3.0, 0.0, 2.5])
expected = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x.clone())
torch.testing.assert_close(actual, expected)
