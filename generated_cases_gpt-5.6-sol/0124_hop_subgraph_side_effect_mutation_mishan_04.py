# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : wrap nonlocal rebinding
# sibling    : nonlocal augmented assignment inside a wrap HOP subgraph

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling under test: nonlocal augmented assignment inside a wrap HOP subgraph.
def execute(use_compile):
    visits = 0

    def fn(x):
        def body(value):
            nonlocal visits
            visits += 1
            return value.cos()

        return wrap(body, x)

    target = torch.compile(fn, backend="eager", fullgraph=True) if use_compile else fn
    output = target(torch.tensor([0.0, 1.0]))
    return output, visits


eager_output, eager_visits = execute(False)
compiled_output, compiled_visits = execute(True)
torch.testing.assert_close(compiled_output, eager_output)
assert compiled_visits == eager_visits, (compiled_visits, eager_visits)
