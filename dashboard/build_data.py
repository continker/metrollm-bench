"""Build dashboard data.json from benchmark results."""

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CASES_FILE = Path(__file__).resolve().parent.parent / "cases" / "cases.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "public" / "data.json"

# v3 runs to include
RUN_FILES = {
    "v3_thinking_run1": ("v3_thinking_run1.json", "v3_thinking_run1_scored.json"),
    "v3_thinking_run2": ("v3_thinking_run2.json", "v3_thinking_run2_scored.json"),
    "v3_thinking_run3": ("v3_thinking_run3.json", "v3_thinking_run3_scored.json"),
    "v3_no_thinking_run1": ("v3_no_thinking_run1.json", "v3_no_thinking_run1_scored.json"),
    "v3_no_thinking_run2": ("v3_no_thinking_run2.json", "v3_no_thinking_run2_scored.json"),
    "v3_no_thinking_run3": ("v3_no_thinking_run3.json", "v3_no_thinking_run3_scored.json"),
}

SCORING_COMPONENTS = [
    "route_correct",
    "fare_correct",
    "tool_calls_correct",
    "no_tool_hallucination",
    "schema_validity",
    "framebook_conformance",
    "disruption_detected",
    "advisory_issued",
    "advisory_content_correct",
    "accessibility_accuracy",
]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def tool_sequence_key(tool_calls: list[dict]) -> str:
    return " → ".join(tc["name"] for tc in tool_calls)


def build_run(run_id: str, raw_file: str, scored_file: str) -> dict:
    raw = load_json(RESULTS_DIR / raw_file)
    scored = load_json(RESULTS_DIR / scored_file)

    thinking = raw.get("thinking", "thinking" in run_id)
    label = run_id.replace("_", " ").title()

    # Index scored results by case_id
    scores_by_id = {s["case_id"]: s for s in scored["scores"]}

    # Index raw results by case_id
    raw_by_id = {r["case_id"]: r for r in raw["results"]}

    scores = {}
    execution = {}
    conversations = {}

    for case_id, r in raw_by_id.items():
        s = scores_by_id.get(case_id, {})
        scores[case_id] = {
            "total": s.get("total", 0),
            "max_possible": s.get("max_possible", 0),
            "breakdown": s.get("breakdown", {}),
        }

        execution[case_id] = {
            "e2e_ms": r.get("e2e_ms"),
            "ttft_ms": r.get("ttft_ms"),
            "input_tokens": r.get("input_tokens"),
            "output_tokens": r.get("output_tokens"),
            "error": r.get("error"),
            "tool_calls": [
                {"name": tc["name"], "arguments": tc.get("arguments", {})}
                for tc in r.get("tool_calls_made", [])
            ],
        }

        conversations[case_id] = {
            "reasoning_content": r.get("reasoning_content"),
            "messages": r.get("messages", []),
        }

    return {
        "run_id": run_id,
        "thinking": thinking,
        "label": label,
        "summary": scored.get("summary", {}),
        "scores": scores,
        "execution": execution,
        "conversations": conversations,
    }


def compute_aggregates(runs: dict, cases: dict) -> dict:
    by_run = {}
    tool_patterns = {}

    for run_id, run in runs.items():
        scores = run["scores"]
        execution = run["execution"]

        # Overall percentage
        total_pts = sum(s["total"] for s in scores.values())
        max_pts = sum(s["max_possible"] for s in scores.values())
        overall_pct = (total_pts / max_pts * 100) if max_pts > 0 else 0

        # By category
        by_category = defaultdict(lambda: {"total": 0, "max": 0, "count": 0})
        for case_id, s in scores.items():
            cat = cases[case_id]["category"]
            by_category[cat]["total"] += s["total"]
            by_category[cat]["max"] += s["max_possible"]
            by_category[cat]["count"] += 1
        by_category = {
            cat: {
                "mean_pct": round(v["total"] / v["max"] * 100, 1) if v["max"] > 0 else 0,
                "count": v["count"],
                "total": v["total"],
                "max": v["max"],
            }
            for cat, v in sorted(by_category.items())
        }

        # By component
        by_component = defaultdict(lambda: {"total": 0, "max": 0, "count": 0})
        for s in scores.values():
            for comp, bd in s["breakdown"].items():
                by_component[comp]["total"] += bd["score"]
                by_component[comp]["max"] += bd["max"]
                by_component[comp]["count"] += 1
        by_component = {
            comp: {
                "mean_pct": round(v["total"] / v["max"] * 100, 1) if v["max"] > 0 else 0,
                "count": v["count"],
            }
            for comp, v in sorted(by_component.items())
        }

        # Latency stats
        latencies = [e["e2e_ms"] for e in execution.values() if e["e2e_ms"] is not None]
        latency_stats = {}
        if latencies:
            latency_stats = {
                "mean": round(statistics.mean(latencies)),
                "median": round(statistics.median(latencies)),
                "p95": round(sorted(latencies)[int(len(latencies) * 0.95)]),
                "min": round(min(latencies)),
                "max": round(max(latencies)),
            }

        # Token stats
        input_tokens = [e["input_tokens"] for e in execution.values() if e["input_tokens"] is not None]
        output_tokens = [e["output_tokens"] for e in execution.values() if e["output_tokens"] is not None]
        token_stats = {}
        if input_tokens:
            token_stats["input_mean"] = round(statistics.mean(input_tokens))
            token_stats["input_total"] = sum(input_tokens)
        if output_tokens:
            token_stats["output_mean"] = round(statistics.mean(output_tokens))
            token_stats["output_total"] = sum(output_tokens)

        by_run[run_id] = {
            "overall_pct": round(overall_pct, 1),
            "by_category": by_category,
            "by_component": by_component,
            "latency": latency_stats,
            "tokens": token_stats,
        }

        # Tool sequence patterns
        patterns = defaultdict(int)
        for e in execution.values():
            seq = tool_sequence_key(e["tool_calls"])
            patterns[seq] += 1
        tool_patterns[run_id] = dict(sorted(patterns.items(), key=lambda x: -x[1]))

    # Thinking vs non-thinking aggregate
    thinking_pcts = [by_run[rid]["overall_pct"] for rid in runs if runs[rid]["thinking"]]
    non_thinking_pcts = [by_run[rid]["overall_pct"] for rid in runs if not runs[rid]["thinking"]]

    thinking_vs = {}
    if thinking_pcts:
        thinking_vs["thinking"] = {
            "mean_pct": round(statistics.mean(thinking_pcts), 1),
            "std_pct": round(statistics.stdev(thinking_pcts), 2) if len(thinking_pcts) > 1 else 0,
        }
    if non_thinking_pcts:
        thinking_vs["non_thinking"] = {
            "mean_pct": round(statistics.mean(non_thinking_pcts), 1),
            "std_pct": round(statistics.stdev(non_thinking_pcts), 2) if len(non_thinking_pcts) > 1 else 0,
        }

    # Per-category thinking vs non-thinking
    for cat_key in set().union(*(by_run[r]["by_category"].keys() for r in by_run)):
        t_pcts = [by_run[r]["by_category"].get(cat_key, {}).get("mean_pct", 0)
                   for r in runs if runs[r]["thinking"]]
        nt_pcts = [by_run[r]["by_category"].get(cat_key, {}).get("mean_pct", 0)
                    for r in runs if not runs[r]["thinking"]]
        thinking_vs[f"cat_{cat_key}_thinking"] = {
            "mean_pct": round(statistics.mean(t_pcts), 1) if t_pcts else 0,
            "std_pct": round(statistics.stdev(t_pcts), 2) if len(t_pcts) > 1 else 0,
        }
        thinking_vs[f"cat_{cat_key}_non_thinking"] = {
            "mean_pct": round(statistics.mean(nt_pcts), 1) if nt_pcts else 0,
            "std_pct": round(statistics.stdev(nt_pcts), 2) if len(nt_pcts) > 1 else 0,
        }

    return {
        "by_run": by_run,
        "thinking_vs_non_thinking": thinking_vs,
        "tool_patterns": tool_patterns,
    }


def main():
    # Load cases
    cases_list = load_json(CASES_FILE)
    cases_by_id = {c["id"]: c for c in cases_list}
    case_order = sorted(cases_by_id.keys(), key=lambda cid: (cases_by_id[cid]["category"], cid))

    # Build case metadata
    cases_meta = {}
    for c in cases_list:
        scoring_comps = list(c.get("scoring", {}).keys())
        cases_meta[c["id"]] = {
            "id": c["id"],
            "category": c["category"],
            "difficulty": c.get("difficulty", "medium"),
            "title": c.get("title", c["id"]),
            "max_possible": sum(c.get("scoring", {}).values()),
            "scoring_components": scoring_comps,
        }

    # Build runs
    runs = {}
    for run_id, (raw_file, scored_file) in RUN_FILES.items():
        raw_path = RESULTS_DIR / raw_file
        scored_path = RESULTS_DIR / scored_file
        if not raw_path.exists() or not scored_path.exists():
            print(f"  Skipping {run_id}: files not found")
            continue
        print(f"  Processing {run_id}...")
        runs[run_id] = build_run(run_id, raw_file, scored_file)

    if not runs:
        print("ERROR: No runs found. Check results/ directory.")
        return

    # Compute aggregates
    aggregates = compute_aggregates(runs, cases_meta)

    # Assemble output
    sample_run = next(iter(runs.values()))
    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "qwen3.5",
            "system": "marta",
            "cases_total": len(cases_by_id),
            "runs_total": len(runs),
        },
        "cases": cases_meta,
        "case_order": case_order,
        "scoring_components": SCORING_COMPONENTS,
        "runs": runs,
        "aggregates": aggregates,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"\nWrote {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print(f"  Runs: {len(runs)}")
    print(f"  Cases: {len(cases_by_id)}")
    print(f"  Components: {len(SCORING_COMPONENTS)}")


if __name__ == "__main__":
    main()
