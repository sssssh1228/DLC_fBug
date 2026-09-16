import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

INPUT_FILE = "issue_summary_v5_9.json"
URL = "****"
LLM_TOKEN = "****"
MODEL = "claude-opus-4-8-thinking"

TARGET_CATEGORIES = [
    "Guard Correctness",
    "Graph Break Correctness",
    "Side-Effect Tracking Fidelity",
    "Semantic Modeling Fidelity-1",
    "Semantic Modeling Fidelity-2"

]

N_MIN = 6
RHO = 1.5
ABS_STRONG_RATIO = 0.10
TOP_K_SEEDS = 4
BATCH_SIZE = 40
ALPHA = 0.05


LIFECYCLE_STAGES = {
    "Guard Correctness": [
        {"name": "Guard Construction Failure",
         "definition": "The guard-building phase itself is wrong: guard generation "
                        "crashes, omits a guard that should exist, or emits an invalid / "
                        "incorrect guard predicate, so the produced guard set is unsound "
                        "from the start."},
        {"name": "Guard Over-Specialization",
         "definition": "Guards are generated too strictly (e.g. a dynamic dimension is "
                        "specialized to a constant, or an over-narrow value/type guard), "
                        "causing spurious recompilations or rejecting inputs that should "
                        "have been accepted by the compiled code."},
        {"name": "Guard Stale Reuse",
         "definition": "A previously built guard or compiled artifact is cached and reused "
                        "when it is no longer valid: the guard wrongly matches after the "
                        "relevant state has changed, so an outdated compiled result is served."},
    ],
    "Graph Break Correctness": [
        {"name": "Graph-Break Decision Mistake",
         "definition": "Dynamo makes a wrong decision about whether / where to break the "
                        "graph: it breaks when it should not, fails to break when it must, "
                        "or breaks at an incorrect bytecode boundary."},
        {"name": "Resume Function Failure",
         "definition": "After a graph break, the generated resume (continuation) function is "
                        "wrong: bytecode reconstruction, live-variable/state restoration, or "
                        "control-flow stitching of the resumed frame is incorrect."},
    ],
    "Side-Effect Tracking Fidelity": [
        {"name": "Mutation Tracking Failure",
         "definition": "An in-place mutation of program state (object attribute, container "
                        "contents, global, etc.) is not correctly recorded or modeled, so the "
                        "side effect is lost or mis-modeled relative to eager execution."},
        {"name": "Replay Ordering Failure",
         "definition": "Tracked side effects are replayed outside the graph in the wrong "
                        "order or at the wrong time, so that the final observable program "
                        "state diverges from eager even though each mutation was recorded."},
    ],
    "Semantic Modeling Fidelity-1": [
        {"name": "Missing VariableTracker",
         "definition": "No suitable VariableTracker type exists to represent the encountered "
                        "Python runtime entity, so Dynamo cannot construct a symbolic proxy "
                        "for it at all."},
        {"name": "Incorrect VariableTracker",
         "definition": "A VariableTracker is constructed, but the wrong VT type is chosen or "
                        "the entity is mapped to an inappropriate symbolic representation."},
        {"name": "Incorrect VariableTracker Initialization",
         "definition": "The correct VariableTracker type is chosen, but its initial symbolic "
                        "state or metadata (e.g. shape, source, contents) is constructed "
                        "incorrectly."},
    ],
    "Semantic Modeling Fidelity-2": [
        {"name": "Missing Operation Handler",
         "definition": "The VariableTracker is correct, but it lacks a handler/method for the "
                        "attempted operation, so the operation cannot be symbolically "
                        "processed."},
        {"name": "Incorrect Dispatch",
         "definition": "The operation on a correctly-constructed VariableTracker is dispatched "
                        "to the wrong handler or produces incorrect in-graph symbolic "
                        "semantics / an incorrect (or missing) FX node."},
    ],
}


def stage_names(root_cause: str) -> list[str]:
    return [s["name"] for s in LIFECYCLE_STAGES.get(root_cause, [])]


def stage_definitions(root_cause: str) -> dict[str, str]:
    return {s["name"]: s["definition"]
            for s in LIFECYCLE_STAGES.get(root_cause, [])}



TEMPERATURE = 0.0

@dataclass
class Issue:
    issue_id: int
    trigger_type: str
    trigger_description: str
    root_cause_category: str
    symptom: str
    summary: str

    def signature_line(self) -> str:
        return (
            f"[{self.issue_id}] trigger={self.trigger_type} | "
            f"symptom={self.symptom} | {self.trigger_description} :: {self.summary}"
        )


def load_issues(path: str) -> list[Issue]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    issues = []
    for r in raw:
        if r.get("root_cause_category") not in TARGET_CATEGORIES:
            continue
        issues.append(Issue(
            issue_id=r["issue_id"],
            trigger_type=r["trigger_type"],
            trigger_description=r.get("trigger_description", ""),
            root_cause_category=r["root_cause_category"],
            symptom=r.get("symptom", ""),
            summary=r.get("summary", ""),
        ))
    # print(f"len of issues:{len(issues)}")
    return issues


def load_all_for_background(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    counts = Counter(r["trigger_type"] for r in raw)
    total = sum(counts.values())
    return {t: c / total for t, c in counts.items()}

def compute_seed_scores(issues_in_domain, background):
    n_s = len(issues_in_domain)
    counts = Counter(i.trigger_type for i in issues_in_domain)
    abs_strong = max(N_MIN, math.ceil(ABS_STRONG_RATIO * n_s))
    rows = []
    for t, n_t in counts.items():
        p_t_given_s = n_t / n_s
        p_t = background.get(t, 1e-9)
        lift = p_t_given_s / p_t if p_t > 0 else 0.0
        score = n_t * math.log(1 + lift)
        is_seed = (n_t >= abs_strong) or (n_t >= N_MIN and lift >= RHO)
        rows.append({
            "trigger_type": t, "absolute": n_t,
            "p_t_given_s": round(p_t_given_s, 4),
            "lift": round(lift, 3), "score": round(score, 3),
            "is_seed": is_seed,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows

def g_test(issues: list[Issue], background: dict[str, float]) -> tuple[float, float]:
    n = len(issues)
    if n == 0:
        return 0.0, 1.0
    obs = Counter(i.trigger_type for i in issues)
    g = 0.0
    for t, o in obs.items():
        e = n * background.get(t, 1e-9)
        if o > 0 and e > 0:
            g += 2 * o * math.log(o / e)
    df = max(len(obs) - 1, 1)
    p = chi2_sf(g, df)
    return g, p


def normalized_entropy(issues: list[Issue]) -> float:
    n = len(issues)
    if n == 0:
        return 0.0
    counts = Counter(i.trigger_type for i in issues)
    m = len(counts)
    if m <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return h / math.log(m)


def chi2_sf(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    return gammaincc(df / 2.0, x / 2.0)


def gammaincc(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        term = summ
        for _ in range(500):
            ap += 1.0
            term *= x / ap
            summ += term
            if abs(term) < abs(summ) * 1e-12:
                break
        p = summ * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - p
    else:
        b = x + 1.0 - a
        c = 1e30
        d = 1.0 / b
        h = d
        for i in range(1, 500):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delt = d * c
            h *= delt
            if abs(delt - 1.0) < 1e-12:
                break
        return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h

from openai import OpenAI
client = OpenAI(
        base_url=URL,
        api_key=LLM_TOKEN
    )


LLM_MAX_RETRIES = 5


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
            last_err = ValueError("LLM 返回空内容")
        except Exception as e:
            last_err = e
        print(f"  [warn] LLM Call Failure({attempt}/{LLM_MAX_RETRIES}): {last_err}")

    raise RuntimeError(
        f"LLM Call {LLM_MAX_RETRIES} Failures: {last_err}") from last_err


def extract_json(text: str) -> Any:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    blob = m.group(1) if m else text
    blob = blob.strip()
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
    raise RuntimeError(
        f"LLM JSON parsing failed {LLM_MAX_RETRIES} times: {last_err}") from last_err


# ============================================================
# 4. Prompts
# ============================================================

SYS_EXTRACT = """You are an expert in PyTorch TorchDynamo internals analyzing bug reports.
You extract *bug patterns*: reusable defect templates sharing a SINGLE root cause.

Rules:
- A bug pattern must NOT span multiple root-cause mechanisms.
- Base every field ONLY on the provided issues. Do NOT invent facts.
- A pattern's "signature" must be a discriminative, checkable condition that lets us
  decide whether an arbitrary other issue belongs to it.
- If the given issues actually contain more than one distinct root cause, output multiple patterns.
Output STRICT JSON only, no prose."""

def prompt_extract(root_cause: str, seed_trigger: str, issues: list[Issue]) -> str:
    lines = "\n".join(i.signature_line() for i in issues)
    return f"""Root-cause mechanism (fixed): {root_cause}
These issues were selected because trigger_type="{seed_trigger}" is statistically
over-represented in this mechanism (a Pattern Discovery Seed).

ISSUES:
{lines}

Task: Propose 1-3 bug pattern(s) that unify these issues under the fixed mechanism.
For EACH issue decide if it fits (some seed issues may be off-topic).

Output JSON:
{{
  "patterns": [
    {{
      "name": "<short pattern name>",
      "root_cause_mechanism": "{root_cause}",
      "signature": "<discriminative checkable condition>",
      "trigger": "<how to trigger this bug>",
      "description": "<1-2 sentences, about 50words>",
      "fix_direction": "<how it is typically fixed>",
      "member_issue_ids": [<ids from above that fit this pattern>]
    }}
  ],
  "unfit_issue_ids": [<seed ids that fit none of the patterns>]
}}"""


SYS_MERGE = """You consolidate candidate bug patterns produced independently from different seeds.
You only see pattern DEFINITIONS (not raw issues).

CRITICAL CONSTRAINTS:
- ALL candidates already share the SAME fixed root-cause mechanism. This shared
  mechanism is a GIVEN PREMISE, NOT a reason to merge or to create a parent.
- NEVER create a parent/umbrella pattern that merely restates the mechanism name
  (e.g. do not output a parent called "Graph Break Correctness").
- Only merge two patterns if they describe the SAME specific defect (near-identical
  signatures / interchangeable). Distinct specific defects MUST stay separate,
  even though they share the mechanism.
- Only create a parent-child link when one pattern is a strictly NARROWER special
  case of ANOTHER SPECIFIC pattern (not of the mechanism). If unsure, keep them as
  separate siblings with parent=null.
- Prefer keeping patterns separate. Merging/parenting is the exception, not the default.

Output STRICT JSON only."""


def prompt_merge(root_cause: str, candidates: list[dict]) -> str:
    body = json.dumps(candidates, ensure_ascii=False, indent=2)
    return f"""Fixed root-cause mechanism (shared by ALL candidates, do NOT use as a merge key): {root_cause}

CANDIDATE PATTERNS:
{body}

Task: Produce a consolidated taxonomy of SPECIFIC bug patterns.
- Merge only near-duplicate patterns describing the same specific defect.
- Keep distinct specific defects as separate patterns (parent=null).
- Set "parent" only if a pattern is a narrower case of ANOTHER specific pattern
  listed here (never the mechanism name itself).

Output JSON:
{{
  "patterns": [
    {{
      "name": "<canonical name>",
      "signature": "<discriminative condition>",
      "trigger": "<how to trigger this bug>",
      "description": "<1-2 sentences>",
      "fix_direction": "<...>",
      "parent": "<another specific pattern name, or null>"
    }}
  ]
}}"""


SYS_ASSIGN = """You assign PyTorch Dynamo bug issues to a FIXED set of bug patterns.
Assign each issue to exactly ONE best-matching pattern, or "RESIDUAL" if none fits.
Prefer the most specific (child) pattern when both a parent and child match.
Base the decision on the pattern signatures. Output STRICT JSON only."""

def prompt_assign(root_cause: str, patterns: list[dict], issues: list[Issue]) -> str:
    ptable = json.dumps(
        [{"name": p["name"], "parent": p.get("parent"),
          "signature": p["signature"]} for p in patterns],
        ensure_ascii=False, indent=2)
    lines = "\n".join(i.signature_line() for i in issues)
    return f"""Root-cause mechanism (fixed): {root_cause}

PATTERNS:
{ptable}

ISSUES:
{lines}

Output JSON:
{{
  "assignments": [
    {{"issue_id": <id>, "pattern": "<pattern name or RESIDUAL>", "confidence": <0-1>}}
  ]
}}"""


SYS_LIFECYCLE = """You classify residual PyTorch Dynamo bug issues by their LIFECYCLE STAGE
within a fixed mechanism (these issues did not form a salient pattern).
Assign each to exactly one given stage, or "Others" if none applies.
Output STRICT JSON only."""

def prompt_lifecycle(root_cause: str, stages: list[dict], issues: list[Issue]) -> str:
    lines = "\n".join(i.signature_line() for i in issues)
    stage_block = "\n".join(
        f"- {s['name']}: {s['definition']}" for s in stages)
    names = [s["name"] for s in stages]
    return f"""Root-cause mechanism: {root_cause}

Assign each issue to exactly one of the following LIFECYCLE STAGES, based on the
stage DEFINITIONS below. If none of the stages genuinely applies, use "Others".
Choose the single best-fitting stage; do NOT invent new stage names.

LIFECYCLE STAGES:
{stage_block}
- Others: none of the above stages applies.

Valid stage names: {json.dumps(names + ["Others"])}

ISSUES:
{lines}

Output JSON:
{{"assignments": [{{"issue_id": <id>, "stage": "<stage or Others>"}}]}}"""


SYS_NONCORE = """You analyze residual PyTorch Dynamo bug issues that could NOT be attributed
to any core Dynamo mechanism (guards / graph break / variable modeling / dispatch).

Your job: determine whether these issues share COMMON DEFECT PATTERNS that are
GENERIC to software engineering and NOT specific to Dynamo's compilation logic.
Examples of such generic categories (use only if supported by the data):
- Documentation and error-message quality
- Environment issues and version inconsistency
- Skip / disable-list logic (issues rooted in TorchDynamo's skipfiles/trace rules
  rather than semantic modeling)

Only propose a category if >=8 issues support it. Assign each issue to exactly one
category, or "Genuinely Unclassifiable" if none fits. Output STRICT JSON only."""


def prompt_noncore(issues: list[Issue]) -> str:
    lines = "\n".join(i.signature_line() for i in issues)
    return f"""These issues are residuals not attributable to any core Dynamo mechanism.

ISSUES:
{lines}

Task: Group them into generic (non-Dynamo-specific) defect patterns.
Output JSON:
{{
  "patterns": [
    {{
      "name": "<generic defect category name>",
      "description": "<1-2 sentences>",
      "member_issue_ids": [<ids>]
    }}
  ],
  "unclassifiable_issue_ids": [<ids fitting none>]
}}"""


def analyze_others(results: dict, all_issues: list[Issue]) -> dict:
    id2issue = {i.issue_id: i for i in all_issues}
    others = [id2issue[iid] for res in results.values()
              for iid, stage in res["residual_lifecycle"].items()
              if stage == "Others" and iid in id2issue]

    print(f"\n{'='*60}\nOthers bug patterns: {len(others)} )\n{'='*60}")
    if not others:
        return {}

    assignments = {}   # issue_id -> generic pattern name
    pattern_defs = {}  # name -> description
    for batch in batches(others, BATCH_SIZE):
        out = call_llm_json(SYS_NONCORE, prompt_noncore(batch))
        for p in out.get("patterns", []):
            pattern_defs.setdefault(p["name"], p.get("description", ""))
            for iid in p.get("member_issue_ids", []):
                assignments[iid] = p["name"]
        for iid in out.get("unclassifiable_issue_ids", []):
            assignments.setdefault(iid, "Genuinely Unclassifiable")

    counts = Counter(assignments.values())
    for name, n in counts.most_common():
        print(f"  [{n:3d}] {name}")
        if name in pattern_defs:
            print(f"        {pattern_defs[name]}")

    return {
        "count": len(others),
        "pattern_definitions": pattern_defs,
        "assignments": assignments,
        "issues": {i.issue_id: {
            "trigger_type": i.trigger_type,
            "trigger_description": i.trigger_description,
            "symptom": i.symptom,
            "summary": i.summary,
        } for i in others},
    }


def batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]



MIN_ABS_TO_CONTINUE = 10

def has_significant_seed(scores):
    if not scores:
        return False
    max_abs = max(r["absolute"] for r in scores)
    if max_abs < MIN_ABS_TO_CONTINUE:
        return False
    return any(r["is_seed"] for r in scores)

def discover_patterns_for_domain(root_cause, domain_issues, background,
                                 max_rounds=1):
    print(f"\n{'='*60}\n机制域: {root_cause}  (共 {len(domain_issues)} 条)\n{'='*60}")

    by_id = {i.issue_id: i for i in domain_issues}
    remaining = list(domain_issues)
    all_patterns = []
    assignments = {}

    round_no = 0
    while remaining and round_no < max_rounds:
        round_no += 1
        print(f"\n{'-'*50}\n[Round {round_no}]  剩余 {len(remaining)} 条\n{'-'*50}")

        scores = compute_seed_scores(remaining, background)
        print("Seed calculation (trigger / abs / lift / score / is_seed):")
        for r in scores:
            print(f"  {r['trigger_type']:18s} abs={r['absolute']:3d} "
                  f"lift={r['lift']:.2f} score={r['score']:.2f} seed={r['is_seed']}")

        if not has_significant_seed(scores):
            print("No Seed -> Stop pattern extraction, turn to lifecycle analysis")
            break

        seeds = [r for r in scores if r["is_seed"]][:TOP_K_SEEDS]

        by_trigger = defaultdict(list)
        for i in remaining:
            by_trigger[i.trigger_type].append(i)

        candidates = []
        for s in seeds:
            t = s["trigger_type"]
            seed_issues = by_trigger[t]
            print(f"\n[Phase A] Extract from Seed '{t}' ({len(seed_issues)} issues) ...")
            out = call_llm_json(SYS_EXTRACT,
                                        prompt_extract(root_cause, t, seed_issues))
            patterns = out.get("patterns", [])
            patterns = [p for p in patterns if isinstance(p, dict) and p.get("name")]
            if not patterns:
                print("    [skip] Find No pattern")
                continue
            best = max(patterns, key=lambda p: len(p.get("member_issue_ids", [])))
            best["_seed"] = t
            candidates.append(best)
            print(f"    Reserve: {best['name']}  "
                  f"(members={len(best.get('member_issue_ids', []))})")

        if not candidates:
            print("Find No pattern -> Stop Iteration")
            break

        print(f"\n[Phase B] Try to merge {len(candidates)} patterns...")
        slim = [{k: c.get(k) for k in
                 ("name", "signature", "trigger", "description", "fix_direction")}
                for c in candidates]
        merged = call_llm_json(SYS_MERGE, prompt_merge(root_cause, slim))
        new_patterns = merged.get("patterns", [])
        for p in new_patterns:
            print(f"    After merge: {p['name']}  parent={p.get('parent')}")

        print(f"\n[Phase C] Review {len(remaining)} issues...")
        round_assign = {}
        for batch in batches(remaining, BATCH_SIZE):
            out = call_llm_json(SYS_ASSIGN,
                                        prompt_assign(root_cause, new_patterns, batch))
            for a in out.get("assignments", []):
                round_assign[a["issue_id"]] = a["pattern"]

        newly_assigned = 0
        for iid, pname in round_assign.items():
            if pname != "RESIDUAL":
                assignments[iid] = pname
                newly_assigned += 1
        all_patterns.extend(new_patterns)
        remaining = [i for i in remaining if i.issue_id not in assignments]

        print(f"    Categorize {newly_assigned} issues in this iteration, leave {len(remaining)} un-categorized")

        if newly_assigned == 0:
            print("No new assignment → Stop iteration")
            break

    lifecycle_result = {}
    if remaining:
        g, p = g_test(remaining, background)
        h = normalized_entropy(remaining)
        print(f"\n[Left {len(remaining)} issues] G={g:.2f}, p={p:.4f}, H_norm={h:.3f}, analyse according to lifecycle")
        stages = LIFECYCLE_STAGES.get(root_cause, [])
        valid_names = set(stage_names(root_cause)) | {"Others"}
        for batch in batches(remaining, BATCH_SIZE):
            out = call_llm_json(SYS_LIFECYCLE,
                                        prompt_lifecycle(root_cause, stages, batch))
            for a in out.get("assignments", []):
                stage = a["stage"] if a["stage"] in valid_names else "Others"
                lifecycle_result[a["issue_id"]] = stage

    return {
        "root_cause": root_cause,
        "rounds": round_no,
        "patterns": all_patterns,
        "pattern_assignments": assignments,
        "residual_lifecycle": lifecycle_result,
    }

def write_pattern_report_json(results: dict[str, Any], all_issues: list[Issue],
                              path: str = f"pattern_report_{MODEL}.json"):
    id2issue = {i.issue_id: i for i in all_issues}

    def issue_dict(iid):
        it = id2issue.get(iid)
        if not it:
            return {"issue_id": iid}
        return {
            "issue_id": it.issue_id,
            "trigger_type": it.trigger_type,
            "trigger_description": it.trigger_description,
            "symptom": it.symptom,
            "summary": it.summary,
        }

    report = {}
    for rc, res in results.items():
        assignments = res["pattern_assignments"]      # {issue_id: pattern_name}
        lifecycle = res["residual_lifecycle"]         # {issue_id: stage}

        # pattern_name -> [issue_id]
        pat2ids = defaultdict(list)
        for iid, pname in assignments.items():
            if pname != "RESIDUAL":
                pat2ids[pname].append(iid)
            
        patterns_out = []
        for p in res["patterns"]:
            ids = sorted(pat2ids.get(p["name"], []))
            patterns_out.append({
                "name": p["name"],
                "parent": p.get("parent"),
                "signature": p.get("signature", ""),
                "trigger": p.get("trigger", ""),
                "description": p.get("description", ""),
                "fix_direction": p.get("fix_direction", ""),
                "count": len(ids),
                "issues": [issue_dict(iid) for iid in ids],
            })


        stage_def_map = stage_definitions(rc)
        stage2ids = defaultdict(list)
        for iid, stage in lifecycle.items():
            stage2ids[stage].append(iid)
        residual_out = []
        for stage, ids in sorted(stage2ids.items(), key=lambda x: -len(x[1])):
            ids = sorted(ids)
            residual_out.append({
                "stage": stage,
                "definition": stage_def_map.get(stage, ""),
                "count": len(ids),
                "issues": [issue_dict(iid) for iid in ids],
            })

        report[rc] = {
            "patterns": patterns_out,
            "residual_lifecycle": residual_out,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Pattern information written to {path}")


def main():
    all_issues = load_issues(INPUT_FILE)
    background = load_all_for_background(INPUT_FILE)
    
    by_domain = defaultdict(list)
    for i in all_issues:
        by_domain[i.root_cause_category].append(i)

    results = {}
    for rc in TARGET_CATEGORIES:
        if rc not in by_domain:
            continue
        results[rc] = discover_patterns_for_domain(rc, by_domain[rc], background)

        with open(f"pattern_discovery_result_{MODEL}.json", "w",
                  encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        write_pattern_report_json(results, all_issues,
                                  f"pattern_report_{MODEL}.json")
        print(f"[Save] Finish analyzing root cause'{rc}', Result saved")

    print(f"\nResult Saved to pattern_discovery_result_{MODEL}.json")


    print("\n" + "=" * 60 + "\nAll Patterns\n" + "=" * 60)
    for rc, res in results.items():
        print(f"\n## {rc}")
        counts = Counter(res["pattern_assignments"].values())
        for p in res["patterns"]:
            n = counts.get(p["name"], 0)
            tag = f"  (parent={p['parent']})" if p.get("parent") else ""
            print(f"  [{n:3d}] {p['name']}{tag}")
        lc = Counter(res["residual_lifecycle"].values())
        for stage, n in lc.most_common():
            print(f"  [{n:3d}] (lifecycle) {stage}")

    others_analysis = analyze_others(results, all_issues)
    with open(f"others_analysis_{MODEL}.json", "w", encoding="utf-8") as f:
        json.dump(others_analysis, f, ensure_ascii=False, indent=2)
    print(f"\nOthers information written to others_analysis_{MODEL}.json")

    write_pattern_report_json(results, all_issues, f"pattern_report_{MODEL}.json")



if __name__ == "__main__":
    main()
