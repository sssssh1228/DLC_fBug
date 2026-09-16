# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Finally side effect after graph break
# sibling    : try/finally mutation surrounding a non-empty graph checkpoint

import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported

# Sibling construct: try/finally side effect surrounding a non-empty graph checkpoint.
def fn(x, state):
    try:
        y = x.square() + 1
        torch._dynamo.graph_break()
        return torch.relu(y - 2)
    finally:
        state["finally_count"] += 1


x = torch.tensor([-2.0, 0.5])
eager_state = {"finally_count": 0}
eager = fn(x, eager_state)

compiled_state = {"finally_count": 0}
torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")(x, compiled_state)
torch.testing.assert_close(compiled, eager)
assert compiled_state == eager_state == {"finally_count": 1}

torch._dynamo.reset()
strict_state = {"finally_count": 0}
strict_fn = torch.compile(fn, backend="eager", fullgraph=True)
try:
    strict_fn(x, strict_state)
except (Unsupported, InternalTorchDynamoError):
    pass
else:
    raise AssertionError("fullgraph=True captured or resumed after a graph break in try/finally")
