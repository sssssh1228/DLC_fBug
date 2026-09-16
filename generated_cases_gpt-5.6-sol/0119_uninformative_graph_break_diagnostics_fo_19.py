# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Skipped warnings function
# sibling    : trace-rules-skipped standard-library warnings.warn call

import warnings

import torch

# Sibling construct: trace-rules-skipped warnings.warn inside tensor-producing code.
warnings.filterwarnings("ignore", message="skip-rule probe")


def fn(x):
    metadata = {"active": x.shape[0] > 0, "elements": x.numel()}
    if metadata["active"]:
        warnings.warn("skip-rule probe", UserWarning)
    flattened = x.reshape(-1)
    return flattened.square() + metadata["elements"]


x = torch.tensor([[1.0, -2.0], [3.5, 0.0]])
expected = fn(x.clone())
compiled = torch.compile(fn, backend="eager")
actual = compiled(x.clone())
torch.testing.assert_close(actual, expected)
