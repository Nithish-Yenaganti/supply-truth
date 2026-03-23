import os
import sys
from pathlib import Path

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.supervisor import app  # Import your compiled LangGraph app

def run_lab_tests():
    test_dir = Path(__file__).resolve().parent
    results = []

    print(f"--- STARTING LAB TESTS in {test_dir} ---\n")

    for filename in os.listdir(test_dir):
        if filename.endswith(".txt"):
            with open(test_dir / filename, "r") as f:
                raw_text = f.read()

            print(f"Testing: {filename}...")
            
            # Trigger the Graph
            final_state = app.invoke({
                "raw_text": raw_text,
                "iterations": 0
            })

            # Record the outcome
            verdict = "PASS SUCCESS" if final_state["critique"]["is_valid"] else "REJECTED FAILURE"
            results.append({
                "file": filename,
                "verdict": verdict,
                "reason": final_state["critique"]["final_decision"]
            })

    # Print Report Card
    print("\n" + "="*30)
    print("      LAB REPORT CARD")
    print("="*30)
    for r in results:
        print(f"{r['file']}: {r['verdict']}")
        print(f"  > Reason: {r['reason']}\n")

if __name__ == "__main__":
    run_lab_tests()
