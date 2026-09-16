# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Changing user-defined length
# sibling    : mutable __len__ protocol

import torch

# Sibling construct: mutable __len__ protocol.
class Window:
    def __init__(self, width):
        self.width = width

    def __len__(self):
        return self.width

    def advance(self):
        self.width += 1


def fn(x, window):
    width = len(window)
    window.advance()
    return x[:width].sum()


def run(callable_fn):
    window = Window(1)
    x = torch.arange(8, dtype=torch.float32)
    outputs = [callable_fn(x, window) for _ in range(5)]
    return outputs, window.width


eager_outputs, eager_width = run(fn)
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend="eager")
compiled_outputs, compiled_width = run(compiled_fn)

assert eager_width == compiled_width
assert len(eager_outputs) == len(compiled_outputs)
for eager, compiled in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(compiled, eager)
