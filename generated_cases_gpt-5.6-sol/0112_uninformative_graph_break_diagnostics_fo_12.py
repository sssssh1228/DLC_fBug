# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Disabled closure with data-dependent branch
# sibling    : torch._dynamo.disable functional wrapper around a closure

import torch

# Sibling construct: torch._dynamo.disable functional wrapper around a closure.
def make_fallback(bias):
    def fallback(x):
        if x.sum().item() >= 0:
            return x + bias
        return x - bias

    return torch._dynamo.disable(fallback, reason="data-dependent closure fallback")


fallback = make_fallback(2.5)


def fn(x):
    return fallback(x * 3).square()


for x in (torch.tensor([1.0, -0.25]), torch.tensor([-2.0, -1.0])):
    eager = fn(x.clone())
    compiled = torch.compile(fn, backend="eager")(x.clone())
    torch.testing.assert_close(compiled, eager)
