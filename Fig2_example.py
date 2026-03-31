import torch
import numpy as np
import torch._dynamo as dynamo
import dis

'''
def model(x):
    y = x + 1
    print("woo")
    return y
'''

global_state = []

def foo(x: torch.Tensor, bias: float):
    # [1. Symbolic Tracing & Guards]
    temp = x * 2 + bias

    # [2. Side-Effect]
    global_state.append("mutation_tracked")

    # [3. Graph Break]
    # .item() forces data access; numpy is unsupported.
    if temp[0].item() > 0:
        np_noise = np.random.rand()
        temp = temp + torch.tensor(np_noise)
    return temp.mean()

# --- Inspection Code ---

input_tensor = torch.randn(4)

print("--- [Python Bytecode] ---")
print(dis.dis(foo))

print("--- [Dynamo Explanation Analysis] ---")
explanation = dynamo.explain(foo)(input_tensor, 1.0)
# print(explanation.__dict__)

print(f"Graph Break Count: {explanation.graph_break_count}")
if explanation.graph_break_count > 0:
    print(f"Break Reason: {explanation.break_reasons}")

opt_fn = torch.compile(foo, backend="inductor")
opt_fn(input_tensor, 1.0)
print(f"\n--- [Side Effect Verification] ---")
print(f"Global State: {global_state}") 

for i, g in enumerate(explanation.graphs):
    print(f"\n=== Graph {i} ===")
    print(g.graph)