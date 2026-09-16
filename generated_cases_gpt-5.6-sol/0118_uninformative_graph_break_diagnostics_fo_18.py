# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Explicit custom graph break
# sibling    : torch._dynamo.graph_break with a custom diagnostic message

import torch

# Sibling construct: torch._dynamo.graph_break carrying an explicit custom message.
def fn(x):
    values = []
    for index in range(2):
        values.append(x + index)
    torch._dynamo.graph_break(msg="manual phase boundary")
    return torch.stack(values).prod(dim=0)


x = torch.tensor([1.0, 2.0, -3.0])
expected = fn(x.clone())
compiled = torch.compile(fn, backend="eager")
actual = compiled(x.clone())
torch.testing.assert_close(actual, expected)

report = torch._dynamo.explain(fn)(x.clone())
reasons = "\n".join(
    str(getattr(item, "reason", item)) for item in report.break_reasons
)
assert "manual phase boundary" in reasons, reasons
