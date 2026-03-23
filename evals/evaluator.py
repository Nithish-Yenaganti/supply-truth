import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.supervisor import app


class SupplyTruthEvaluator:
    def __init__(self, eval_file: str, output_path: str):
        self.eval_file = Path(eval_file)
        self.output_path = Path(output_path)
        self.results: list[dict[str, Any]] = []
        self.stats = {
            "total_cases": 0,
            "perfect_matches": 0,
            "partial_matches": 0,
            "critical_failures": 0,
            "total_latency": 0.0,
            "avg_latency": 0.0,
        }

    def _normalize_actual(self, final_state: dict[str, Any]) -> dict[str, Any]:
        data = final_state.get("extracted_data") or {}
        items = data.get("items") or []
        first_item = items[0] if items else {}

        normalized = dict(data)
        if "sku" not in normalized and first_item.get("sku") is not None:
            normalized["sku"] = first_item.get("sku")
        if "quantity" not in normalized and first_item.get("quantity") is not None:
            normalized["quantity"] = first_item.get("quantity")
        return normalized

    def calculate_score(self, expected: dict[str, Any], actual: dict[str, Any]) -> float:
        """Simple field-level accuracy over fields present in expected."""
        if not expected:
            return 0.0

        matched = 0
        for key, exp_value in expected.items():
            act_value = actual.get(key)

            if key in {"quantity", "price"}:
                try:
                    if float(act_value) == float(exp_value):
                        matched += 1
                except (TypeError, ValueError):
                    pass
                continue

            if key == "eta":
                # Accept exact match or same date prefix for ISO timestamps.
                exp_str = str(exp_value)
                act_str = str(act_value) if act_value is not None else ""
                if act_str == exp_str or act_str.startswith(exp_str):
                    matched += 1
                continue

            if act_value is not None and str(act_value).strip().lower() == str(exp_value).strip().lower():
                matched += 1

        return round(matched / len(expected), 2)

    def run_benchmarks(self, limit: int | None = None):
        print(f"--- SupplyTruth Audit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        if not self.eval_file.exists():
            print(f"Error: {self.eval_file} not found.")
            return

        lines = [line.strip() for line in self.eval_file.read_text().splitlines() if line.strip()]
        if limit is not None:
            lines = lines[:limit]

        for idx, line in enumerate(lines, start=1):
            self.stats["total_cases"] += 1
            case = json.loads(line)
            case_name = case.get("expected", {}).get("shipment_id", f"case_{idx}")
            print(f"Testing {idx}/{len(lines)}: {case_name}...", end=" ", flush=True)

            start_time = time.time()
            try:
                final_state = app.invoke({"raw_text": case["input"], "iterations": 0})
                latency = time.time() - start_time
                self.stats["total_latency"] += latency

                critique = final_state.get("critique", {})
                actual = self._normalize_actual(final_state)
                expected = case.get("expected", {})
                score = self.calculate_score(expected, actual)

                if score == 1.0:
                    self.stats["perfect_matches"] += 1
                    status = "PERFECT"
                elif score >= 0.5:
                    self.stats["partial_matches"] += 1
                    status = "PARTIAL"
                else:
                    self.stats["critical_failures"] += 1
                    status = "FAIL"

                print(f"{status} ({latency:.2f}s)")
                self.results.append(
                    {
                        "case_id": idx,
                        "input": case.get("input"),
                        "expected": expected,
                        "actual": actual,
                        "critique": critique,
                        "score": score,
                        "latency_sec": round(latency, 2),
                    }
                )
            except Exception as exc:
                self.stats["critical_failures"] += 1
                print(f"CRASHED: {exc}")
                self.results.append(
                    {
                        "case_id": idx,
                        "input": case.get("input"),
                        "expected": case.get("expected", {}),
                        "actual": {},
                        "critique": {"is_valid": False, "final_decision": str(exc)},
                        "score": 0.0,
                        "latency_sec": round(time.time() - start_time, 2),
                    }
                )

        self.finalize_report()

    def finalize_report(self):
        total = self.stats["total_cases"]
        accuracy = 0.0
        if total > 0:
            self.stats["avg_latency"] = self.stats["total_latency"] / total
            accuracy = (self.stats["perfect_matches"] / total) * 100

        print("\n" + "=" * 50)
        print("         SUPPLYTRUTH PERFORMANCE DASHBOARD")
        print("=" * 50)
        print(f"Overall Accuracy:    {accuracy:.2f}%")
        print(f"Avg Latency:         {self.stats['avg_latency']:.2f} sec/request")
        print(f"Perfect Matches:     {self.stats['perfect_matches']}")
        print(f"Partial Matches:     {self.stats['partial_matches']}")
        print(f"Critical Failures:   {self.stats['critical_failures']}")
        print("-" * 50)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stats": self.stats,
            "generated_at": datetime.now().isoformat(),
            "results": self.results,
        }
        self.output_path.write_text(json.dumps(payload, indent=2))
        print(f"Full Audit Log saved to: {self.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SupplyTruth evaluator against jsonl dataset")
    parser.add_argument("--eval-file", default="evals/data_set.json1", help="Path to JSONL eval dataset")
    parser.add_argument("--output", default="data/evals/benchmark_report.json", help="Output report path")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N cases")
    args = parser.parse_args()

    evaluator = SupplyTruthEvaluator(eval_file=args.eval_file, output_path=args.output)
    evaluator.run_benchmarks(limit=args.limit)
