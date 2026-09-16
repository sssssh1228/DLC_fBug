# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Channels-last clone preservation
# sibling    : torch.clone with preserve_format on a channels-last tensor

import torch

# Sibling under test: channels-last stride propagation through torch.clone.
class PreserveFormatClone:
    def __call__(self, tensor):
        return torch.clone(tensor, memory_format=torch.preserve_format)


x = torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last)
fn = PreserveFormatClone()

eager = fn(x)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(x)

assert actual.device == eager.device
assert actual.dtype == eager.dtype
assert actual.shape == eager.shape
assert actual.stride() == eager.stride()
assert actual.is_contiguous(memory_format=torch.channels_last)
torch.testing.assert_close(actual, eager)
