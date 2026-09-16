# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Reflected power dispatch
# sibling    : pow() fallback dispatch to __rpow__

import torch

# Sibling under test: pow() fallback dispatch to __rpow__.
class TensorExponent:
    def __init__(self, value):
        self.value = value

    def __rpow__(self, base):
        return self.value.square() + base


def fn(x):
    return pow(3, TensorExponent(x))


x = torch.tensor([2.0, -4.0, 0.5])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
