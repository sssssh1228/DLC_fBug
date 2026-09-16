# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Granular Break Override
# sibling    : nested error_on_graph_break(False) region overriding an outer strict context

import torch

# Sibling construct: nested error_on_graph_break(False) region permitting one local resume.
def fn(x):
    y = x.sin() + 1
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()
        y = y * 3
    return y.cos() - 2


x = torch.linspace(-2.0, 2.0, 7)
eager_result = fn(x.clone())

torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend='eager')
compiled_result = compiled_fn(x.clone())
torch.testing.assert_close(compiled_result, eager_result)

torch._dynamo.reset()
strict_compiled_fn = torch.compile(fn, backend='eager')
with torch._dynamo.error_on_graph_break(True):
    strict_context_result = strict_compiled_fn(x.clone())
torch.testing.assert_close(strict_context_result, eager_result)
