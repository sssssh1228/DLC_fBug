# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Missing or incorrect BuiltinVariable handler for Python builtin semantics
# title      : Strict Zip Iteration
# sibling    : zip() with the strict keyword argument over tensor tuples

import torch

# Sibling construct: zip(..., strict=True) over equal-length tensor tuples.
def fn(a, b, c, scale):
    left = (a, b, c)
    right = (scale, scale + 1, scale + 2)
    products = [value * factor for value, factor in zip(left, right, strict=True)]
    return torch.stack(products)


a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
c = torch.tensor([5.0, 6.0])
scale = torch.tensor(2.0)
eager = fn(a, b, c, scale)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(a, b, c, scale)
torch.testing.assert_close(compiled, eager)
