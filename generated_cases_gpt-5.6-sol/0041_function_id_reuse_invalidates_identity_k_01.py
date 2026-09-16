# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Polyfill registry after nested-function collection
# sibling    : torch.compiler.substitute_in_graph original-function registry

import gc
import torch

# Sibling construct: substitute_in_graph registration for a nested function.
def try_reuse(dead_id, factory, attempts=10000):
    for _ in range(attempts):
        candidates = [factory() for _ in range(16)]
        for candidate in candidates:
            if id(candidate) == dead_id:
                return candidate
        del candidates, candidate
    return factory()


def make_original():
    def original(x, scale=2):
        return x * scale
    return original


def make_candidate():
    def candidate(x, scale=2):
        return x - scale
    return candidate


original = make_original()
dead_id = id(original)

@torch.compiler.substitute_in_graph(original)
def polyfill(x, scale=2):
    return x * scale

del original
gc.collect()
candidate = try_reuse(dead_id, make_candidate)


def workload(x):
    return candidate(x, scale=3) + 1


x = torch.tensor([4.0, -2.0])
eager = workload(x)
compiled = torch.compile(workload, backend="eager", fullgraph=True)(x)
torch.testing.assert_close(compiled, eager)
