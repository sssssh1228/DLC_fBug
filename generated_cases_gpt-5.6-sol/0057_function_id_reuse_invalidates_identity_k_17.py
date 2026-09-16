# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Recycled nonstrict-trace identity
# sibling    : torch.compiler.nonstrict_trace

import torch

# Sibling construct: nonstrict_trace callable registry entry.
def make_old():
    def old(x):
        return torch.cos(x)

    return old


def make_candidate():
    return lambda x: x.add_(3.0).square()


def find_reused_id(target_id, factory):
    for _ in range(4096):
        batch = [factory() for _ in range(128)]
        for value in batch:
            if id(value) == target_id:
                return value
    raise RuntimeError("CPython did not reuse the function id")


nonstrict_trace = getattr(torch.compiler, "nonstrict_trace", None)
if nonstrict_trace is None:
    nonstrict_trace = torch._dynamo.nonstrict_trace

victim = nonstrict_trace(make_old())
stale_id = id(victim)
del victim
candidate = find_reused_id(stale_id, make_candidate)
assert id(candidate) == stale_id


def run(x):
    result = candidate(x)
    return result, x


x_eager = torch.tensor([-2.0, 0.0, 4.0])
eager_result, eager_mutated = run(x_eager)
x_compiled = torch.tensor([-2.0, 0.0, 4.0])
compiled_result, compiled_mutated = torch.compile(
    run, backend="eager", fullgraph=True
)(x_compiled)
torch.testing.assert_close(compiled_result, eager_result)
torch.testing.assert_close(compiled_mutated, eager_mutated)
