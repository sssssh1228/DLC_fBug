# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Contextlib cleanup across graph break
# sibling    : generator-based contextlib context manager with local side effects

import contextlib
import torch
from torch._dynamo.exc import Unsupported

# Sibling construct: generator-based contextlib context manager with local side effects.
@contextlib.contextmanager
def record_scope(records):
    records.append("enter")
    try:
        yield
    finally:
        records.append("exit")


def fn(x):
    records = []
    with record_scope(records):
        y = x + 1
        torch._dynamo.graph_break()
        y = y * 2
    return y + len(records)


x = torch.tensor([1.0, 2.0])
eager = fn(x)
torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")
actual = compiled(x)
torch.testing.assert_close(actual, eager)

torch._dynamo.reset()
strict = torch.compile(fn, backend="eager", fullgraph=True)
try:
    strict(x)
except Unsupported:
    pass
else:
    raise AssertionError("fullgraph=True silently resumed after a contextlib graph break")
