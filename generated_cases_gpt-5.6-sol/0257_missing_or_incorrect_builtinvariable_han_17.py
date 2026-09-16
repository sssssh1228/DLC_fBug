# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Round with precision dispatch
# sibling    : round() dispatch to __round__ with ndigits

import torch


class RoundedBox:
    def __init__(self, value):
        self.value = value

    def __round__(self, ndigits=None):
        if ndigits is None:
            return torch.round(self.value)
        scale = 10 ** ndigits
        return torch.round(self.value * scale) / scale


def fn(x, digits):
    # Sibling under test: round() dispatch to __round__ with ndigits.
    box = RoundedBox(x)
    return round(box, digits)


x = torch.tensor([1.234, -5.678, 9.876])
expected = fn(x.clone(), 2)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x.clone(), 2)
torch.testing.assert_close(actual, expected)
