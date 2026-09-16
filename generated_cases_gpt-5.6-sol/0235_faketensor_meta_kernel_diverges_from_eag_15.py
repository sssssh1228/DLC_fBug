# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : FakeTensor/meta kernel diverges from eager semantics
# title      : Lazy conjugate view metadata
# sibling    : Tensor.conj lazy conjugate-bit and transposed-stride propagation

import torch

# Sibling under test: Tensor.conj lazy conjugate-bit metadata propagation.
def fn(payload):
    value = payload["value"]
    return value.transpose(0, 1).conj()


real = torch.arange(12, dtype=torch.float32).reshape(3, 4)
x = torch.complex(real, real + 0.5)
payload = {"value": x}

expected = fn(payload)
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
actual = compiled_fn(payload)

torch.testing.assert_close(actual, expected)
assert actual.shape == expected.shape
assert actual.stride() == expected.stride()
assert actual.storage_offset() == expected.storage_offset()
assert actual.dtype == expected.dtype
assert actual.device == expected.device
assert actual.is_conj() == expected.is_conj()
