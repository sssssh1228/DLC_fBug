import json
from openai import OpenAI
import os
import re

GITHUB_TOKEN = "****"
LLM_TOKEN = "****"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_parse_llm_output(text):
    """
    Try to extract JSON from LLM output safely.
    Never crash.
    """
    try:
        # first try direct parse
        return json.loads(text)
    except Exception:
        pass

    # try extract from code block
    # m = re.search(r"```json\s*(.*?)```", text, re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # fallback: extract first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    return None

def summarize_with_llm(issue_info, torch_tax_path, variabletracker_tax_path, second_stage=False):
    torch_taxonomy = load_json(torch_tax_path)
    variabletracker_taxonomy = load_json(variabletracker_tax_path)

    user_prompt = f"""
You are an expert in PyTorch TorchDynamo internals.
Analyze the following TorchDynamo GitHub issue and produce a structured classification.

1. Modeled Entity: Select exactly **one** types from the list below that are most relevant to triggering this bug
{variabletracker_taxonomy}
2. Trigger Description (≤10 words): Describe how a Python language construct or usage pattern triggers the bug
Keep the description abstract and structural (e.g., "capturing closure variable in loop", "overriding magic method on tuple subclass")
3. Root Cause Category: Select exactly **one** category from the taxonomy below:
{torch_taxonomy}

Read the definition of each root cause. You may refer to the classification instruction when unclear:
(1) First compilation is correct, bug defects appear on recompilation? Yes -> Guard Correctness
(2) Defect occurs in graph break's partitioning or resume logic? Yes -> Graph Break Correctness
(3) Mutation of program state is invloved in the defect? Yes -> Side-Effect Tracking Fidelity
(4) Otherwise, is the defect related to tensor computation, objects, attribute resolution -> Semantic Modeling Fidelity

4. Symptom: Select exactly **one** from: crash, timeout, inconsistency
5. Summary (≤30 words): Briefly describe the interaction between the Python-level behavior and the failed Dynamo mechanism


Example:
for this issue:
{{
"title": "torch.compile failed to handle a custom __delattr__ method correctly",
"body": "### Describe the bug\n\ntorch.compile fails to correctly handle classes with a custom `__delattr__` method. Specifically, when a class overrides `__delattr__` to block deletion of certain attributes, the behavior is not preserved under compilation.\n\nMRE:\n```python\nimport torch\n\nclass MyObject:\n    def __init__(self, val):\n        self.val = val\n\n    def __delattr__(self, attr):\n        if attr == \"val\":\n            print(f\"Cannot delete attribute '{{attr}}'!\")\n        else:\n            super().__delattr__(attr)\n\n@torch.compile(fullgraph=False, backend=\"eager\")\ndef test(input_tensor):\n    instance_a = MyObject(1)\n    instance_b = MyObject(2)\n\n    del instance_a.val\n    del instance_b.val\n    exists_a = hasattr(instance_a, 'val')\n    exists_b = hasattr(instance_b, 'val')\n\n    return input_tensor + 1, exists_a, exists_b\n\n# Expected output: (tensor([2.]), True, True) since 'val' deletion is prevented\n# Actual output: (tensor([2.]), False, False)\nprint(test(torch.ones(1)))\n```\n\nAlso, if we dont use `@torch.compile`, this error does not appear. This suggests that the cumtom `__delattr__` is bypassed or not respected during graph tracing or ahead-of-time compilation.\n\n### Error logs\n\nTerminal output:\n\n```\n(tensor([2.]), False, False)\n```\n\nAnd `Cannot delete attribute '{{attr}}'!` is not printed.\n\n### Versions\n\npython 3.10.14\npytorch 2.4.0\n\ncc @ezyang @gchanan @zou3519 @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @chenyang78 @amjames",
      "comments": [
        "@XinyiYuan @StrongerXi  created a pr for this issue, please review it https://github.com/pytorch/pytorch/pull/150899\n"
      ]

}}

Expected output:
{{
"modeled_entity": "...",
"trigger_description": "overide a magic function in a module",
"root_cause_category": "...",
"symptom": "inconsistency",
"summary": "..."
}}

The Issue that you should analyze:
------------------
{issue_info}
"""
    user_prompt_second =f"""
You are an expert in PyTorch TorchDynamo internals.
The root cause of this issue has already been identified as related to Semantic Modeling in TorchDynamo.

In TorchDynamo, Semantic Modeling uses VariableTracker objects to symbolically model Python runtime objects and their behaviors during tracing, and ultimately construct a correct FX Graph while preserving the relevant Python semantics outside the graph.

Your task is to further classify the issue into one of the following two stages of Semantic Modeling.

Stage 1 — Variable Construction Fidelity:
Whether TorchDynamo can correctly identify a Python runtime entity and construct/map it to the appropriate VariableTracker representation.
This stage includes:
- No appropriate VariableTracker exists for the runtime entity.
- The wrong VariableTracker type is selected.
- The runtime object is mapped to an inappropriate symbolic representation.
- The initial symbolic state or metadata of the VariableTracker is incorrectly constructed.

Stage 2 — Operation Dispatch Fidelity:
Given that the relevant VariableTracker has already been correctly constructed, whether TorchDynamo correctly handles an operation on that VariableTracker and produces the correct in-graph symbolic semantics / FX representation.
This stage includes:
- The VariableTracker exists and is correct, but the operation is dispatched to the wrong handler/method.
- The VariableTracker lacks the appropriate method for handling the operation.
- The operation produces an incorrect or missing FX node.
- The operation is incorrectly lowered, represented, or constant-folded within the FX graph.
- The local symbolic semantics of the operation are incorrect.
Note that Stage 2 is NOT limited to stateless operations. An operation may update the local symbolic state of a VariableTracker (e.g., iterator position or container contents) and still belong to Stage 2 if that state transition is part of the operation's own symbolic semantics.

Expected output:
{{
"stage": "Stage 1" | "Stage 2",
"reason": Explain why this stage is the root cause, ~50 words,
}}

The Issue that you should analyze:
------------------
{issue_info}
    
    """
    
    if second_stage:
        user_prompt = user_prompt_second
    
    # print(user_prompt_second)
    client = OpenAI(
        base_url="****",
        api_key=LLM_TOKEN
    )
    
    response = client.chat.completions.create(
        model="claude-opus-4-6-thinking",
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    result_text = response.choices[0].message.content.strip()
    # print(result_text)
    return result_text

if __name__ == "__main__":
    issues = load_json("issues.json")
    # issues = issues[:20]
    summary_results = load_json("issue_summary_v5_8_6.json")
    summary_map = {item["issue_id"]: item for item in summary_results}
    
    for target_issue in issues:
        target_issue_id = target_issue.get("number")
        
        output_dir = "summarized_result_v5_8_28"
        os.makedirs("summarized_result_v5_8_28", exist_ok=True)
        output_filename = os.path.join(output_dir, f"{target_issue_id}_summary_2.json")
        
        if os.path.exists(output_filename):
            print(f"Skipping {target_issue_id}: file already exists.")
            continue
        
        issue_summary = summary_map.get(target_issue_id)
        
        if issue_summary is None:
            print(f"Warning: issue #{target_issue_id} not found in issue_summary_v5_1.json")
            continue
        
        root_category = issue_summary.get("root_cause_category")
        if root_category == "Semantic Modeling Fidelity":
            print(f"Issue #{target_issue_id} requires second-stage classification")
            raw_output = summarize_with_llm(target_issue, torch_tax_path="taxonomy_v5_1.json", variabletracker_tax_path="variableTracker.json", second_stage=True)
        else:
            continue
        
        bug_pattern = safe_parse_llm_output(raw_output)
        
        final_result = {
            "issue_id": target_issue_id,
            "root_cause_category": root_category,
            "semantic_stage_analysis": bug_pattern
        }
        
        with open(output_filename,"w",encoding="utf-8") as f:
            json.dump(final_result,f,ensure_ascii=False,indent=4)


        print(f"[Saved] {output_filename}")

        
'''
if __name__ == "__main__":
    issues = load_json("issues.json")
    # issues = issues[:20]
    # print(issues)
    for target_issue in issues:
        # example_issue_id = 150765 # Our issue
        target_issue_id = target_issue.get("number")
        
        output_dir = "summarized_result_v5_8_6_gpt"
        os.makedirs("summarized_result_v5_8_6_gpt", exist_ok=True)
        output_filename = os.path.join(output_dir, f"{target_issue_id}_summary.json")
        
        if os.path.exists(output_filename):
            print(f"Skipping {target_issue_id}: file already exists.")
            continue
        
        
        print("=============================================")
        print(f"Analyzing issue #{target_issue_id} ...")
        # print(issue_info)
        
        # -------- Summarize --------
        raw_output = summarize_with_llm(target_issue, torch_tax_path="taxonomy_v5_1.json", variabletracker_tax_path="variableTracker.json", second_stage=False)

        print("\n================ RAW LLM OUTPUT ================\n")
        print(raw_output)
        
        bug_pattern = safe_parse_llm_output(raw_output)
        print("\n================ PARSED RESULT ================\n")
        print(bug_pattern)
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(bug_pattern, f, ensure_ascii=False, indent=4)
        print(f"[Saved] {output_filename}")
        
        # break
'''