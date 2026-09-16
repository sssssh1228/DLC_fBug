# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Checkpoint set mutation
# sibling    : set.add

import torch
from torch.utils.checkpoint import checkpoint


def make_case():
    visited = set()

    def body(x):
        # Sibling construct under test: set.add.
        visited.add("checkpoint_body")
        return x - 1

    def fn(x):
        return checkpoint(body, x, use_reentrant=False)

    def snapshot():
        return frozenset(visited)

    return fn, snapshot


eager_fn, eager_snapshot = make_case()
eager_out = eager_fn(torch.tensor(5.0))
eager_state = eager_snapshot()

torch._dynamo.reset()
compiled_fn_source, compiled_snapshot = make_case()
compiled_fn = torch.compile(compiled_fn_source, backend="eager")
compiled_out = compiled_fn(torch.tensor(5.0))
compiled_state = compiled_snapshot()

torch.testing.assert_close(compiled_out, eager_out)
assert compiled_state == eager_state, (compiled_state, eager_state)
