# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Custom reason on disabled method
# sibling    : torch.compiler.disable decorator with a user-supplied reason

import torch

# Sibling construct: torch.compiler.disable on an instance method with a custom reason.
class Calibrator:
    @torch.compiler.disable(reason="calibration boundary")
    def adjust(self, x, scale):
        return x.mul(scale).add(1.0)


calibrator = Calibrator()


def fn(x, scale):
    before = x.sin()
    adjusted = calibrator.adjust(before, scale)
    return adjusted.cos(), adjusted.sum()


x = torch.tensor([-1.0, 0.25, 2.0])
expected = fn(x.clone(), 1.5)
compiled = torch.compile(fn, backend="eager")
actual = compiled(x.clone(), 1.5)
torch.testing.assert_close(actual[0], expected[0])
torch.testing.assert_close(actual[1], expected[1])

report = torch._dynamo.explain(fn)(x.clone(), 1.5)
reasons = "\n".join(
    str(getattr(item, "reason", item)) for item in report.break_reasons
)
assert "calibration boundary" in reasons, reasons
