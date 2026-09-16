# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Recycled disable-wrapper identity
# sibling    : torch._dynamo.disable

import torch

# Sibling construct: disable decorator wrapper.
def old(x):
    return torch.exp(x)


def make_candidate():
    def candidate(x, factor=3.0):
        return torch.relu(x) / factor

    return candidate


def find_reused_id(target_id, factory):
    for _ in range(4096):
        batch = [factory() for _ in range(128)]
        for value in batch:
            if id(value) == target_id:
                return value
    raise RuntimeError("CPython did not reuse the function id")


victim = torch._dynamo.disable(old)
del old
stale_id = id(victim)
del victim
candidate = find_reused_id(stale_id, make_candidate)
assert id(candidate) == stale_id


def run(x):
    values = (candidate(x), candidate(-x, factor=2.0))
    return values[0] - values[1]


x = torch.tensor([-4.0, -0.5, 1.5])
eager = run(x.clone())
compiled = torch.compile(run, backend="eager", fullgraph=True)(x.clone())
torch.testing.assert_close(compiled, eager)
