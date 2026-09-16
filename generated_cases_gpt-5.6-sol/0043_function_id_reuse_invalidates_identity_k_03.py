# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Forbidden static method does not taint a replacement
# sibling    : torch._dynamo.forbid_in_graph function marking

import gc
import torch

# Sibling construct: forbid_in_graph marking on a static method.
def try_reuse(dead_id, factory, attempts=10000):
    for _ in range(attempts):
        candidates = [factory() for _ in range(16)]
        for candidate in candidates:
            if id(candidate) == dead_id:
                return candidate
        del candidates, candidate
    return factory()


def make_candidate():
    def candidate(x):
        return x.square() + 4
    return candidate


class Holder:
    @staticmethod
    @torch._dynamo.forbid_in_graph
    def blocked(x):
        return x.neg()


dead_id = id(Holder.blocked)
del Holder
gc.collect()
candidate = try_reuse(dead_id, make_candidate)


def workload(x):
    return candidate(x) / 2


x = torch.tensor([-3.0, 5.0])
eager = workload(x)
compiled = torch.compile(workload, backend="eager", fullgraph=True)(x)
torch.testing.assert_close(compiled, eager)
