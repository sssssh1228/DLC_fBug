# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Checkpoint list extension
# sibling    : list.extend

import torch
from torch.utils.checkpoint import checkpoint


def make_case():
    events = []

    def body(x):
        # Sibling construct under test: list.extend.
        events.extend(["entered", "completed"])
        return x * 2

    def fn(x):
        return checkpoint(body, x, use_reentrant=False)

    def snapshot():
        return tuple(events)

    return fn, snapshot


eager_fn, eager_snapshot = make_case()
eager_out = eager_fn(torch.tensor(3.0))
eager_state = eager_snapshot()

torch._dynamo.reset()
compiled_fn_source, compiled_snapshot = make_case()
compiled_fn = torch.compile(compiled_fn_source, backend="eager")
compiled_out = compiled_fn(torch.tensor(3.0))
compiled_state = compiled_snapshot()

torch.testing.assert_close(compiled_out, eager_out)
assert compiled_state == eager_state, (compiled_state, eager_state)
