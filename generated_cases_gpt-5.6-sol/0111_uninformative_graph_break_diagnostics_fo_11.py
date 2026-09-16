# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Disabled instance method with custom reason
# sibling    : torch.compiler.disable decorator applied to an instance method

import torch

# Sibling construct: torch.compiler.disable decorator on an instance method.
class AffineTransform:
    @torch.compiler.disable(reason="instance-method fallback")
    def shift(self, x, amount):
        return x + amount


transform = AffineTransform()


def fn(x):
    doubled = x * 2
    return transform.shift(doubled, 3) - 1


x = torch.tensor([1.0, -2.0, 4.0])
eager = fn(x.clone())
compiled = torch.compile(fn, backend="eager")(x.clone())
torch.testing.assert_close(compiled, eager)
