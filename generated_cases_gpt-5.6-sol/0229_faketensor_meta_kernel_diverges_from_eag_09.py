# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Empty-strided storage layout
# sibling    : torch.empty_strided with a nonstandard dense stride

import torch

# Sibling under test: torch.empty_strided with a nonstandard dense stride.
def fn(x):
    result = torch.empty_strided(
        (2, 3),
        (1, 2),
        dtype=x.dtype,
        device=x.device,
    )
    result.copy_(x)
    return result, result.stride(), result.storage_offset()

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())

torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1] == eager[1], (compiled[1], eager[1])
assert compiled[2] == eager[2], (compiled[2], eager[2])
assert compiled[0].dtype == eager[0].dtype
assert compiled[0].device == eager[0].device
