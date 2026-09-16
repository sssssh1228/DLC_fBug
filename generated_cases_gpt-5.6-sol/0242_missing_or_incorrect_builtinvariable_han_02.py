# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Round With Precision
# sibling    : round() with ndigits dispatch to a user-defined __round__ method

import torch

# Sibling construct: round(value, ndigits) dispatch through __round__.
class Quantized:
    def __init__(self, value):
        self.value = value

    def __round__(self, ndigits=None):
        scale = 1 if ndigits is None else 10 ** ndigits
        return torch.round(self.value * scale) / scale


def fn(x):
    return round(Quantized(x), 2)


x = torch.tensor([1.234, -5.678, 9.995])
eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x)
torch.testing.assert_close(compiled, eager)
