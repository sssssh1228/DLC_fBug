# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Recycled disallow-in-graph identity
# sibling    : torch._dynamo.disallow_in_graph

import torch

# Sibling construct: disallow_in_graph registry entry.
def make_old():
    captured = 11

    def old(x):
        return x + captured

    return old


def make_candidate():
    scale = 2.5

    def candidate(x):
        return torch.sin(x) * scale

    return candidate


def find_reused_id(target_id, factory):
    for _ in range(4096):
        batch = [factory() for _ in range(128)]
        for value in batch:
            if id(value) == target_id:
                return value
    raise RuntimeError("CPython did not reuse the function id")


victim = make_old()
torch._dynamo.allow_in_graph(victim)
torch._dynamo.disallow_in_graph(victim)
stale_id = id(victim)
del victim
candidate = find_reused_id(stale_id, make_candidate)
assert id(candidate) == stale_id


def run(x):
    return candidate(x) + x.square()


x = torch.tensor([-1.0, 0.25, 2.0])
eager = run(x.clone())
compiled = torch.compile(run, backend="eager", fullgraph=True)(x.clone())
torch.testing.assert_close(compiled, eager)
