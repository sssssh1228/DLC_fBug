# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : wrap list.extend replay
# sibling    : list.extend inside a wrap HOP subgraph

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling under test: list.extend inside a wrap HOP subgraph.
def execute(use_compile):
    events = []

    def fn(x):
        def body(value):
            events.extend(["entered", "completed"])
            return value.sin() + 1

        return wrap(body, x)

    target = torch.compile(fn, backend="eager", fullgraph=True) if use_compile else fn
    output = target(torch.tensor([0.25, -0.5]))
    return output, tuple(events)


eager_output, eager_events = execute(False)
compiled_output, compiled_events = execute(True)
torch.testing.assert_close(compiled_output, eager_output)
assert compiled_events == eager_events, (compiled_events, eager_events)
