# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Round dunder argument forwarding
# sibling    : round() dispatch to __round__ with default and ndigits arguments

import torch

# Sibling under test: round() dispatch to __round__ with default and ndigits arguments.
class RoundedValue:
    def __init__(self, value):
        self.value = value

    def __round__(self, ndigits=None):
        if ndigits is None:
            return self.value - 1
        return self.value + ndigits


def fn(x):
    default_result = round(RoundedValue(x))
    explicit_result = round(RoundedValue(x), ndigits=2)
    return default_result + explicit_result


x = torch.tensor([1.25, -2.75, 3.5])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
