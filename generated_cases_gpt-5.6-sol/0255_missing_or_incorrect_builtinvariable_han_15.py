# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Strict zip iteration
# sibling    : zip() with the strict keyword over tensor iterables

import torch

# Sibling under test: zip() with the strict keyword over tensor iterables.
def fn(x, y):
    combined = []
    for left, right in zip(x.unbind(0), y.unbind(0), strict=True):
        combined.append(left - right)
    return torch.stack(combined)


x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = torch.tensor([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]])
eager = fn(x.clone(), y.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone(), y.clone())
torch.testing.assert_close(compiled, eager)
