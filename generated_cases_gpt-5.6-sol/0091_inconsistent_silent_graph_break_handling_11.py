# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Generator context manager graph break
# sibling    : contextlib.contextmanager enter/exit side effects across a graph break

import contextlib
import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported

# Sibling construct: generator-based context manager with enter/exit side effects.
@contextlib.contextmanager
def tagged_region(events):
    events.append("enter")
    try:
        yield 3
    finally:
        events.append("exit")


def fn(x):
    events = []
    with tagged_region(events) as scale:
        y = x * scale
        torch._dynamo.graph_break()
        y = y + 1
    return y, tuple(events)


x = torch.tensor([1.0, -2.0])
eager_value, eager_events = fn(x)

torch._dynamo.reset()
compiled_value, compiled_events = torch.compile(fn, backend="eager")(x)
torch.testing.assert_close(compiled_value, eager_value)
assert compiled_events == eager_events == ("enter", "exit")

torch._dynamo.reset()
strict_fn = torch.compile(fn, backend="eager", fullgraph=True)
try:
    strict_fn(x)
except (Unsupported, InternalTorchDynamoError):
    pass
else:
    raise AssertionError("fullgraph=True silently accepted a graph break inside a contextlib context manager")
