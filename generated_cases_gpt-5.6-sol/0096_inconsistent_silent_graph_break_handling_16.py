# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Context Manager Side Effects
# sibling    : generator-based contextlib context manager spanning a graph break

import contextlib
import torch

# Sibling construct: generator-based contextlib context manager with enter/exit side effects.
@contextlib.contextmanager
def record_region(events):
    events.append('enter')
    try:
        yield
    finally:
        events.append('exit')


def fn(x, events):
    with record_region(events):
        y = x + 1
        torch._dynamo.graph_break()
        y = torch.relu(y)
    return y * 2


def strict_errors():
    return tuple(
        cls
        for cls in (
            getattr(torch._dynamo.exc, 'Unsupported', None),
            getattr(torch._dynamo.exc, 'InternalTorchDynamoError', None),
        )
        if cls is not None
    )


x = torch.tensor([-2.0, 1.0, 3.0])
eager_events = []
eager_result = fn(x.clone(), eager_events)

torch._dynamo.reset()
compiled_events = []
compiled_fn = torch.compile(fn, backend='eager')
compiled_result = compiled_fn(x.clone(), compiled_events)
torch.testing.assert_close(compiled_result, eager_result)
assert compiled_events == eager_events == ['enter', 'exit']

torch._dynamo.reset()
strict_fn = torch.compile(fn, backend='eager', fullgraph=True)
try:
    strict_fn(x.clone(), [])
except strict_errors():
    pass
else:
    raise AssertionError('fullgraph=True silently resumed across the contextlib graph break')
