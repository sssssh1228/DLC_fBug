# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Reverse Tensor Sequence
# sibling    : reversed() applied to a Python list containing tensors

import torch

# Sibling construct: reversed() over a list of tensor values.
def fn(a, b, c):
    values = [a + 1, b * 2, c - 3]
    return torch.stack(list(reversed(values)))


a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
c = torch.tensor([5.0, 6.0])
eager = fn(a, b, c)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(a, b, c)
torch.testing.assert_close(compiled, eager)
