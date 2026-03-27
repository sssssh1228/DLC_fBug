from openai import OpenAI
import pandas as pd
import os

LLM_TOKEN ="..."

# ── Config ──────────────────────────────────────────────────────────────────
EXCEL_PATH  = "bug_pattern.xlsx"
OUTPUT_DIR  = "generated_cases"
N_CASES     = 10
# ────────────────────────────────────────────────────────────────────────────


def generate_with_llm(name: str, description: str, N: int = 10) -> str:
    user_prompt = f"""
You are an expert in compiler testing and fuzzing, specializing in TorchDynamo — the frontend compiler of PyTorch's torch.compile stack. Your goal is to detect whether TorchDynamo faithfully models Python semantics — specifically, whether functions compiled via torch.compile produce timeouts, crashes, or inconsistent results compared to eager execution.
Your task follows the history-driven analogical generalization methodology: given a known bug pattern, identify the underlying semantic category the defective function belongs to, then generate test cases targeting sibling members of that category rather than the known defective function itself. (e.g., __add__ → binary arithmetic dunder methods → __mul__, __sub__, __truediv__, etc.) The goal is to probe whether the fix was root-cause generic or merely function-specific.

Target Bug Pattern: 
Name: {name} 
Description: {description}

Generation Instructions:
- Each test case must be minimal and self-contained
- Run both eager and compiled (backend="eager") modes, then assert consistency
- Add a one-line comment per test case naming the sibling method under test

Generate {N} test cases.
"""

    client = OpenAI(
        base_url="...",
        api_key=LLM_TOKEN,
    )

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    result_text = response.choices[0].message.content
    return result_text


def load_patterns(excel_path: str):
    """Read A=name, B=description from the first sheet, no header assumed."""
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1])
    df = df.dropna(subset=[0])
    return df.values.tolist()          # [[name, description], ...]


def save_result(output_dir: str, index: int, result: str) -> str:
    """Strip markdown fences and save as {index}.py."""
    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[-1]           # drop opening ``` line
        result = result.rsplit("```", 1)[0].strip()  # drop closing ```
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{index}.py")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(result)
    return filepath


def main():
    print(f"Loading patterns from: {EXCEL_PATH}")
    patterns = load_patterns(EXCEL_PATH)
    print(f"Found {len(patterns)} bug patterns.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined_log_path = os.path.join(OUTPUT_DIR, "_ALL_RESULTS.txt")

    with open(combined_log_path, "w", encoding="utf-8") as combined:
        for idx, (name, description) in enumerate(patterns, start=1):
            name        = str(name).strip()
            description = str(description).strip()

            print(f"[{idx}/{len(patterns)}] Generating for: {name}")

            try:
                result = generate_with_llm(name, description, N=N_CASES)
            except Exception as e:
                result = f"[ERROR] {e}"
                print(f"  ✗ Error: {e}")

            # ── Per-pattern file ──────────────────────────────────────────
            saved_path = save_result(OUTPUT_DIR, idx, result)
            print(f"  ✓ Saved → {saved_path}")

            # ── Append to combined log ────────────────────────────────────
            combined.write(f"{'=' * 60}\n")
            combined.write(f"[{idx}] Pattern : {name}\n")
            combined.write(f"{'=' * 60}\n\n")
            combined.write(result)
            combined.write("\n\n")
            combined.flush()

            print("=" * 45)

    print(f"\nAll done. Results saved in: {OUTPUT_DIR}")
    print(f"Combined log : {combined_log_path}")


if __name__ == "__main__":
    main()