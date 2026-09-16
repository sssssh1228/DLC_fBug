# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Skipped logging side effect
# sibling    : logging.Logger.warning trace-rules-skipped side-effecting method

import logging
import torch

# Sibling construct: logging.Logger.warning exercises a skipped side-effecting method.
class RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


logger = logging.getLogger("torchdynamo_graph_break_sibling")
logger.setLevel(logging.WARNING)
logger.propagate = False
handler = RecordingHandler()
logger.handlers = [handler]


def fn(x, label):
    y = x.relu()
    logger.warning("processed:%s", label)
    return y + 4


value = torch.tensor([-2.0, 0.5, 3.0])
handler.messages.clear()
eager = fn(value.clone(), "eager")
eager_messages = list(handler.messages)

handler.messages.clear()
compiled = torch.compile(fn, backend="eager")
actual = compiled(value.clone(), "compiled")
compiled_messages = list(handler.messages)

torch.testing.assert_close(actual, eager)
assert eager_messages == ["processed:eager"]
assert compiled_messages == ["processed:compiled"]
