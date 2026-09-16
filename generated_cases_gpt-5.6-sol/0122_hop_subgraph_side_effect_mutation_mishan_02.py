# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : wrap dictionary pop replay
# sibling    : dict.pop inside a wrap HOP subgraph

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling under test: dict.pop inside a wrap HOP subgraph.
def execute(use_compile):
    state = {"offset": 3}

    def fn(x):
        def body(value):
            offset = state.pop("offset")
            return value + offset

        return wrap(body, x)

    target = torch.compile(fn, backend="eager", fullgraph=True) if use_compile else fn
    output = target(torch.tensor([1.0, 2.0]))
    return output, dict(state)


eager_output, eager_state = execute(False)
compiled_output, compiled_state = execute(True)
torch.testing.assert_close(compiled_output, eager_output)
assert compiled_state == eager_state, (compiled_state, eager_state)
