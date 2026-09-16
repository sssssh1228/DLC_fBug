# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Reversed sequence fallback
# sibling    : reversed() fallback through __len__ and __getitem__

import torch

# Sibling under test: reversed() fallback to the sequence protocol.
class TensorSequence:
    def __init__(self, value):
        self.value = value

    def __len__(self):
        return self.value.shape[0]

    def __getitem__(self, index):
        return self.value[index]


def fn(x):
    rows = [row * (index + 1) for index, row in enumerate(reversed(TensorSequence(x)))]
    return torch.stack(rows)


x = torch.arange(12.0).reshape(4, 3)
eager = fn(x)
compiled = torch.compile(fn, backend="eager", fullgraph=True)(x)
assert torch.equal(compiled, eager), (compiled, eager)
