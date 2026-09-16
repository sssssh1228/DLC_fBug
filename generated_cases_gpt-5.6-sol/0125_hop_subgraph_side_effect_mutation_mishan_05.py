# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : wrap module attribute deletion
# sibling    : module attribute deletion inside a wrap HOP subgraph

import torch
from torch import nn
from torch._higher_order_ops.wrap import wrap


class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = "present"


# Sibling under test: delattr on an nn.Module inside a wrap HOP subgraph.
def execute(use_compile):
    holder = Holder()

    def fn(x):
        def body(value):
            delattr(holder, "marker")
            return value * 2

        return wrap(body, x)

    target = torch.compile(fn, backend="eager", fullgraph=True) if use_compile else fn
    output = target(torch.tensor([1.5, -2.0]))
    return output, hasattr(holder, "marker")


eager_output, eager_has_marker = execute(False)
compiled_output, compiled_has_marker = execute(True)
torch.testing.assert_close(compiled_output, eager_output)
assert compiled_has_marker == eager_has_marker, (compiled_has_marker, eager_has_marker)
