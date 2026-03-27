import json
from openai import OpenAI
import os
import re

LLM_TOKEN ="..."

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def classify_with_llm(issue_info, torch_taxonomy):
    user_prompt = f"""
You are analyzing a PyTorch compilation bug related to the torch.compile / TorchDynamo pipeline.
Your task is to describe this issue as a structured triplet:[Trigger, Root Cause, Bug Symptom]

This triplet should explain how a high-level Python language construct or usage pattern triggers a failure in TorchDynamo’s compilation or optimization logic, ultimately leading to an observable incorrect behavior at runtime.

For a given PyTorch issue description, you must:
1. Describe the Trigger (≤10 words) to explain how a Python language construct or usage pattern triggers the bug.
Keep the description abstract and structural (e.g., “capturing closure variable in loop”).
Do not mention any specific functions, APIs, or concrete operations (avoid words like “delete”, “create”).
2. Identify the primary root cause category from the provided Root Cause taxonomy. The root cause must reflect which compiler mechanism or modeling assumption fails.
3. Select one Bug Symptom from {{crash, timeout, inconsistency}}.
crash: runtime exception, assertion failure, or compiler abort
timeout: non-termination or excessive compilation/execution time
inconsistency: semantic mismatch between eager and compiled execution

4. Provide a concise Summary (≤20 words)
Briefly describe the interaction between the Python-level behavior and the failed optimization strategy.

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
"trigger": "overide a magic function in a module",
"root_cause": "Symbolic execution",
"symptom": "inconsistency",
"summary": "Custom attribute mutation semantics are ignored due to incomplete side-effect modeling during compilation."
}}


The Issue that you should analyze:
------------------
{issue_info}

Root Cause taxonomy:
------------------
{torch_taxonomy}

"""
    client = OpenAI(
        base_url="...",
        api_key=LLM_TOKEN
    )
    
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    result_text = response.choices[0].message.content.strip()
    return result_text


if __name__ == "__main__":
    issues = load_json("raw_issue_data.json")

    torch_tax_path = "dynamo_mech.json"
    torch_taxonomy = load_json(torch_tax_path)

    all_results = []

    for target_issue in issues:
        if target_issue is None:
            continue
        
        issue_info = target_issue.get("issue_info")
        target_issue_id = issue_info.get("number")

        print("=============================================")
        print(f"Analyzing issue #{target_issue_id} ...")

        # -------- Summarize --------
        analysis_json = classify_with_llm(issue_info, torch_taxonomy)

        print(f"--------Bug pattern--------\n{analysis_json}")
        print("=============================================")

        all_results.append({
            "issue_id": target_issue_id,
            "analysis": analysis_json
        })

    output_path = "llm_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_results)} results to {output_path}")