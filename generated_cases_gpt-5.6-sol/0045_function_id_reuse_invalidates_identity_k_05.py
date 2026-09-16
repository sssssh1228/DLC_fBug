# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Disable wrapper identity is not inherited
# sibling    : torch.compiler.disable wrapper classification

import gc
import torch

# Sibling construct: disable decoration of a function with positional-only arguments.
def try_reuse(dead_id, factory, attempts=10000):
    for _ in range(attempts):
        candidates = [factory() for _ in range(16)]
        for candidate in candidates:
            if id(candidate) == dead_id:
                return candidate
        del candidates, candidate
    return factory()


def make_disabled_target():
    def target(x, /):
        return x.cos()
    return target


def make_candidate():
    def candidate(x, /):
        return x.relu() * 5
    return candidate


target = make_disabled_target()
disabled = torch.compiler.disable(target)
dead_id = id(disabled)
del disabled, target
gc.collect()
candidate = try_reuse(dead_id, make_candidate)


def workload(x):
    return candidate(x) - 1


x = torch.tensor([-2.0, 3.0])
eager = workload(x)
compiled = torch.compile(workload, backend="eager", fullgraph=True)(x)
torch.testing.assert_close(compiled, eager)
