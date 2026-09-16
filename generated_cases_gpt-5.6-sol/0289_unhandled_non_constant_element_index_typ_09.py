# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Unhandled non-constant element/index type in list & Size operations
# title      : List repetition with scalar tensor
# sibling    : list repetition using a scalar-tensor repeat count

import torch

# Sibling under test: list repetition using a scalar-tensor repeat count.
def fn(x, repeats):
    values = [x.sum()] * repeats
    return torch.stack(values).sum()

x = torch.randn(5)
repeats = torch.tensor(3, dtype=torch.int64)
eager = fn(x, repeats)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x, repeats)
torch.testing.assert_close(compiled, eager)
