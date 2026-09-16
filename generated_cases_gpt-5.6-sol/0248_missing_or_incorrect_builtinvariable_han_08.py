# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Round with ndigits
# sibling    : round() forwarding an explicit ndigits argument to __round__

import torch

# Sibling under test: round() forwarding ndigits through __round__.
class DecimalTensor:
    def __init__(self, value):
        self.value = value

    def __round__(self, ndigits=None):
        if ndigits is None:
            return torch.round(self.value)
        scale = 10 ** ndigits
        return torch.round(self.value * scale) / scale


def fn(x):
    wrapped = DecimalTensor(x)
    return round(wrapped, 2), round(wrapped)


x = torch.tensor([1.234, -2.675, 8.499])
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)(x)
assert len(compiled) == len(eager)
assert all(torch.equal(actual, expected) for actual, expected in zip(compiled, eager)), (compiled, eager)
