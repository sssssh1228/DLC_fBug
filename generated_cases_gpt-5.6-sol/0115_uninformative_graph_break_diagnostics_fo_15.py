# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Trace-rules-skipped standard-library call
# sibling    : trace-rules-skipped C function called from a loop

import time

import torch

# Sibling construct: trace-rules-skipped time.sleep calls inside a Python loop.
def fn(x):
    result = x
    for scale in (2.0, 3.0):
        time.sleep(0)
        result = result * scale + 1
    return result


x = torch.tensor([1.0, -2.0, 0.5])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager")
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
