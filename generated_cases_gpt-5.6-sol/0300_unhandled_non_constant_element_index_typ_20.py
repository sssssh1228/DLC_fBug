# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : Scalar tensor in Size construction
# sibling    : torch.Size construction from a symbolic dimension and scalar tensor dimension

import torch

# Sibling under test: torch.Size construction with a scalar-tensor element.
def fn(x, width):
    output_shape = torch.Size((x.shape[0], width))
    return torch.ones(output_shape, dtype=x.dtype, device=x.device) * x.sum()

compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
for x, width in ((torch.randn(2, 4), torch.tensor(3)), (torch.randn(5, 4), torch.tensor(2))):
    eager = fn(x, width)
    compiled = compiled_fn(x, width)
    torch.testing.assert_close(compiled, eager)
