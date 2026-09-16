import json
import os
import re
import time
from typing import Any

# ============================================================
# Config
# ============================================================
INPUT_FILE = "pattern_report_claude-opus-4-8-thinking.json"
URL = "****"
LLM_TOKEN = "****"
MODEL = "gpt-5.6-sol"
OUTPUT_DIR = f"generated_cases_{MODEL}"

CALLS_PER_PATTERN = 4
CASES_PER_CALL = 5
TEMPERATURE = 0.8
SLEEP_SEC = 1
LLM_MAX_RETRIES = 5

from openai import OpenAI
client = OpenAI(base_url=URL, api_key=LLM_TOKEN)

def call_llm(system: str, user: str) -> str:
    last_err = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                reasoning_effort="high",
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content
            last_err = ValueError("LLM Returns NULL")
        except Exception as e:
            last_err = e
        print(f"  [warn] LLM Call Failure({attempt}/{LLM_MAX_RETRIES}): {last_err}")
    raise RuntimeError(f"LLM Call {LLM_MAX_RETRIES} Failures: {last_err}") from last_err


def extract_json(text: str) -> Any:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    blob = (m.group(1) if m else text).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        for pat in (r"\[.*\]", r"\{.*\}"):
            mm = re.search(pat, blob, re.DOTALL)
            if mm:
                return json.loads(mm.group(0))
        raise


def call_llm_json(system: str, user: str) -> Any:
    last_err = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        text = call_llm(system, user)
        try:
            return extract_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(f"  [warn] JSON parsing failed({attempt}/{LLM_MAX_RETRIES}): {e}")
    raise RuntimeError(f"LLM JSON parsing failed {LLM_MAX_RETRIES} times: {last_err}") from last_err


# ============================================================
# Prompt（history-driven analogical generalization）
# ============================================================
SYS_GEN = """You are an expert in compiler testing and fuzzing, specializing in
TorchDynamo — the frontend compiler of PyTorch's torch.compile stack. Your goal is
to detect whether TorchDynamo faithfully models Python semantics — specifically,
whether functions compiled via torch.compile produce timeouts, crashes, or
inconsistent results compared to eager execution.

You follow the history-driven analogical generalization methodology: given a known
bug pattern, identify the underlying semantic category the defective behavior belongs
to, then generate test cases targeting SIBLING members of that category rather than
the known defective function itself (e.g., __add__ -> binary arithmetic dunder
methods -> __mul__, __sub__, __truediv__). The goal is to probe whether the fix was
root-cause generic or merely function-specific.

Output STRICT JSON only, no prose."""


def prompt_gen(root_cause: str, pattern: dict, n: int) -> str:
    return f"""Root-cause mechanism: {root_cause}

Target Bug Pattern:
- Name: {pattern.get('name','')}
- Signature: {pattern.get('signature','')}
- Description: {pattern.get('description','')}
- Typical fix direction: {pattern.get('fix_direction','')}

Generation Instructions:
- Each test case must be minimal and self-contained (imports + code).
- Run both eager and compiled (backend="eager") modes, then assert consistency.
- Add a one-line comment per test case naming the sibling method/construct under test.
- Vary the Python constructs across the {n} cases (do not just rename variables).
- Each "code" must be a COMPLETE runnable python script.

Generate {n} test cases.

Output JSON:
{{
  "test_cases": [
    {{
      "title": "<short title>",
      "sibling_under_test": "<the sibling construct this case probes>",
      "code": "<complete self-contained python source as a single string>"
    }}
  ]
}}"""


def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return s[:maxlen] or "pattern"



def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        defs = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []
    global_idx = 0

    for rc, v in defs.items():
        patterns = v.get("patterns", [])
        print(f"\n{'='*60}\nRoot Cause: {rc}  ({len(patterns)} 个 pattern)\n{'='*60}")

        for p in patterns:
            name = p.get("name", "unnamed")
            slug = slugify(name)
            print(f"\n[Pattern] {name}")
            case_no = 0

            for call_i in range(1, CALLS_PER_PATTERN + 1):
                out = call_llm_json(SYS_GEN, prompt_gen(rc, p, CASES_PER_CALL))
                batch = out.get("test_cases", []) if isinstance(out, dict) else []
                batch = [c for c in batch if isinstance(c, dict) and c.get("code")]

                for c in batch:
                    case_no += 1
                    global_idx += 1
                    fname = f"{global_idx:04d}_{slug}_{case_no:02d}.py"
                    fpath = os.path.join(OUTPUT_DIR, fname)

                    code = c["code"].strip()
                    # remove markdown ```
                    if code.startswith("```"):
                        code = code.split("\n", 1)[-1]
                        code = code.rsplit("```", 1)[0].strip()

                    header = (
                        f"# -*- pattern-testcase -*-\n"
                        f"# root_cause : {rc}\n"
                        f"# pattern    : {name}\n"
                        f"# title      : {c.get('title','')}\n"
                        f"# sibling    : {c.get('sibling_under_test','')}\n\n"
                    )
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(header + code + "\n")

                    manifest.append({
                        "file": fname,
                        "root_cause": rc,
                        "pattern": name,
                        "title": c.get("title", ""),
                        "sibling_under_test": c.get("sibling_under_test", ""),
                    })

                print(f"    call {call_i}/{CALLS_PER_PATTERN}: +{len(batch)} 个")
                time.sleep(SLEEP_SEC)

            with open(os.path.join(OUTPUT_DIR, "_manifest.json"),
                      "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"    [Save] {name}: {case_no} .py")

    print(f"\nFinish: {global_idx} Test cases → {OUTPUT_DIR}/")
    print(f"List: {OUTPUT_DIR}/_manifest.json")


if __name__ == "__main__":
    main()
