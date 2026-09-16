# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Ternary power dunder dispatch
# sibling    : pow() with a modulus dispatching all arguments to __pow__

import torch

# Sibling under test: ternary pow() dispatch through __pow__.
class ModularTensor:
    def __init__(self, value):
        self.value = value

    def __pow__(self, exponent, modulus=None):
        result = self.value ** exponent
        return result if modulus is None else torch.remainder(result, modulus)


def fn(x):
    return pow(ModularTensor(x), 3, 5)


x = torch.tensor([2, 3, 4], dtype=torch.int64)
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)(x)
assert torch.equal(compiled, eager), (compiled, eager)
