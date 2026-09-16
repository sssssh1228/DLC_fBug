# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Checkpoint nonlocal rebinding
# sibling    : nonlocal augmented assignment

import torch
from torch.utils.checkpoint import checkpoint


def make_case():
    invocation_count = 0

    def body(x):
        nonlocal invocation_count
        # Sibling construct under test: nonlocal augmented assignment.
        invocation_count += 1
        return x + invocation_count

    def fn(x):
        return checkpoint(body, x, use_reentrant=False)

    def snapshot():
        return invocation_count

    return fn, snapshot


eager_fn, eager_snapshot = make_case()
eager_out = eager_fn(torch.tensor(8.0))
eager_state = eager_snapshot()

torch._dynamo.reset()
compiled_fn_source, compiled_snapshot = make_case()
compiled_fn = torch.compile(compiled_fn_source, backend="eager")
compiled_out = compiled_fn(torch.tensor(8.0))
compiled_state = compiled_snapshot()

torch.testing.assert_close(compiled_out, eager_out)
assert compiled_state == eager_state, (compiled_state, eager_state)
