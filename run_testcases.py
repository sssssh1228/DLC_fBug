import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def run_one(path: str, timeout: int) -> dict:
    start = time.time()
    status = "ok"
    stdout, stderr, returncode = "", "", None
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        if returncode != 0:
            status = "error"
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[TIMEOUT after {timeout}s]"
    except Exception as e:
        status = "crash"
        stderr = f"[runner exception] {e}"
    elapsed = time.time() - start
    return {
        "file": os.path.basename(path),
        "status": status,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 2),
        "stdout": stdout,
        "stderr": stderr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    folder = args.folder
    py_files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.endswith(".py") and not f.startswith("_")
    )
    print(f"Find {len(py_files)} .py testcases, timeout={args.timeout}s")

    out_prefix = args.out or os.path.join(folder, "_run_results")
    txt_path = out_prefix + ".txt"
    json_path = out_prefix + ".json"

    results = []
    summary = {"ok": 0, "error": 0, "timeout": 0, "crash": 0}

    with open(txt_path, "w", encoding="utf-8") as log:
        log.write(f"Run started: {datetime.now().isoformat()}\n")
        log.write(f"Folder: {folder}  |  {len(py_files)} files  |  "
                  f"timeout={args.timeout}s\n\n")
        log.flush()

        for idx, path in enumerate(py_files, 1):
            name = os.path.basename(path)
            print(f"[{idx}/{len(py_files)}] {name} ...", end=" ", flush=True)
            r = run_one(path, args.timeout)
            results.append(r)
            summary[r["status"]] = summary.get(r["status"], 0) + 1
            print(f"{r['status']}  ({r['elapsed_sec']}s)")

            log.write("=" * 70 + "\n")
            log.write(f"[{idx}] {name}\n")
            log.write(f"status={r['status']}  returncode={r['returncode']}  "
                      f"elapsed={r['elapsed_sec']}s\n")
            log.write("-" * 70 + "\n")
            log.write("[STDOUT]\n" + (r["stdout"] or "") + "\n")
            log.write("[STDERR]\n" + (r["stderr"] or "") + "\n\n")
            log.flush()

            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"summary": summary, "results": results},
                          jf, ensure_ascii=False, indent=2)

    print("\n===== All Results =====")
    for k, v in summary.items():
        print(f"  {k:8s}: {v}")
    print(f"\nTXT Log: {txt_path}")
    print(f"JSON log: {json_path}")


if __name__ == "__main__":
    main()
