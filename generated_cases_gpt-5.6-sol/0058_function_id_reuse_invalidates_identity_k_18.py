# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Recycled forbid-in-graph marker
# sibling    : torch._dynamo.forbid_in_graph

import torch

# Sibling construct: forbid_in_graph function marker.
def make_old():
    def forbidden(x, *, offset=1.0):
        return x + offset

    return forbidden


def make_candidate():
    def candidate(x, *, lower=-0.5, upper=1.5):
        return torch.clamp(x, min=lower, max=upper)

    return candidate


def find_reused_id(target_id, factory):
    for _ in range(4096):
        batch = [factory() for _ in range(128)]
        for value in batch:
            if id(value) == target_id:
                return value
    raise RuntimeError("CPython did not reuse the function id")


victim = make_old()
torch._dynamo.forbid_in_graph(victim)
stale_id = id(victim)
del victim
candidate = find_reused_id(stale_id, make_candidate)
assert id(candidate) == stale_id


def run(x):
    return candidate(x, lower=-1.0, upper=2.0) * 4.0


x = torch.tensor([-3.0, -0.25, 3.0])
eager = run(x.clone())
compiled = torch.compile(run, backend="eager", fullgraph=True)(x.clone())
torch.testing.assert_close(compiled, eager)
