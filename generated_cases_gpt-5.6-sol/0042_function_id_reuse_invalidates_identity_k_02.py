# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Disallow registry after lambda collection
# sibling    : torch._dynamo.disallow_in_graph callable classification

import gc
import torch

# Sibling construct: disallow_in_graph registration for a lambda.
def try_reuse(dead_id, factory, attempts=10000):
    for _ in range(attempts):
        candidates = [factory() for _ in range(16)]
        for candidate in candidates:
            if id(candidate) == dead_id:
                return candidate
        del candidates, candidate
    return factory()


def make_blocked():
    return lambda x: x.sin()


def make_candidate():
    return lambda x: x * 3 - 2


blocked = make_blocked()
torch._dynamo.allow_in_graph(blocked)
torch._dynamo.disallow_in_graph(blocked)
dead_id = id(blocked)
del blocked
gc.collect()
candidate = try_reuse(dead_id, make_candidate)


def workload(x):
    return candidate(x) + x


x = torch.tensor([1.0, 2.0])
eager = workload(x)
compiled = torch.compile(workload, backend="eager", fullgraph=True)(x)
torch.testing.assert_close(compiled, eager)
