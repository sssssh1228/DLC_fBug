# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Unfold view storage metadata
# sibling    : Tensor.unfold stride and storage-offset computation on a narrowed view

import torch

# Sibling under test: Tensor.unfold view stride and storage-offset metadata.
class WindowModule(torch.nn.Module):
    def forward(self, x):
        narrowed = x.narrow(0, 1, 10)
        return narrowed.unfold(0, size=3, step=2)


module = WindowModule()
x = torch.arange(12, dtype=torch.float32)

expected = module(x)
compiled_module = torch.compile(module, backend="eager", fullgraph=True)
actual = compiled_module(x)

torch.testing.assert_close(actual, expected)
assert actual.shape == expected.shape
assert actual.stride() == expected.stride()
assert actual.storage_offset() == expected.storage_offset()
assert actual.dtype == expected.dtype
assert actual.device == expected.device
