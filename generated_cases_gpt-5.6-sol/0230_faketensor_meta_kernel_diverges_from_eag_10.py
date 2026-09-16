# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Pooling memory-format propagation
# sibling    : torch.nn.functional.max_pool2d channels-last stride propagation

import torch
import torch.nn.functional as F

# Sibling under test: torch.nn.functional.max_pool2d channels-last stride propagation.
def fn(x):
    channels_last = x.contiguous(memory_format=torch.channels_last)
    result = F.max_pool2d(channels_last, kernel_size=2, stride=1)
    is_channels_last = result.is_contiguous(memory_format=torch.channels_last)
    return result, result.stride(), is_channels_last

x = torch.arange(2 * 3 * 5 * 6, dtype=torch.float32).reshape(2, 3, 5, 6)
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
compiled = compiled_fn(x.clone())

torch.testing.assert_close(compiled[0], eager[0])
assert compiled[1] == eager[1], (compiled[1], eager[1])
assert compiled[2] == eager[2], (compiled[2], eager[2])
assert compiled[0].dtype == eager[0].dtype
assert compiled[0].device == eager[0].device
