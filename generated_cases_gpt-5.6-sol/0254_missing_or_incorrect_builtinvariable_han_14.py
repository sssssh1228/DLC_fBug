# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Symbolic custom length
# sibling    : len() dispatch to __len__ returning a symbolic dimension

import torch

# Sibling under test: len() dispatch to __len__ returning a symbolic dimension.
class TensorSized:
    def __init__(self, value):
        self.value = value

    def __len__(self):
        return self.value.shape[0]


def fn(x):
    size = len(TensorSized(x))
    offsets = torch.arange(size, dtype=x.dtype, device=x.device)
    return x + offsets


x = torch.tensor([10.0, 20.0, 30.0, 40.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
