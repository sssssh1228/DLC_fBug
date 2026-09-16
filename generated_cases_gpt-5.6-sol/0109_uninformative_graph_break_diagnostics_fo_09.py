# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Skipped introspection helper
# sibling    : inspect.signature trace-rules-skipped standard-library function

import inspect
import torch

# Sibling construct: inspect.signature exercises a trace-rules-skipped introspection function.
def target(alpha, beta=1, *, gamma=2):
    return alpha + beta + gamma


def fn(x):
    signature = inspect.signature(target)
    positional = sum(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in signature.parameters.values()
    )
    return x.cos() + positional


value = torch.tensor([-0.75, 0.0, 1.25])
eager = fn(value.clone())
compiled = torch.compile(fn, backend="eager")
actual = compiled(value.clone())
torch.testing.assert_close(actual, eager)
