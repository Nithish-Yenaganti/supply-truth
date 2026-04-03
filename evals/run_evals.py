import os
import sys
from pathlib import Path
from langsmith import Client
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.supervisor import app

# 1. Setup Cache (Optional/Hybrid)
if os.getenv("CLEAN_EVAL") != "true":
    set_llm_cache(SQLiteCache(database_path="evals/cache.db"))

client = Client()

# --- CUSTOM EVALUATORS ---

def eval_json_exact(run, example, **kwargs) -> dict:
    """Checks if the total JSON is an exact match to the reference."""
    predicted = run.outputs.get("extracted_data", {})
    expected = example.outputs.get("expected", {})
    
    # Simple boolean check
    score = 1.0 if predicted == expected else 0.0
    return {"key": "json_exact", "score": score}

def eval_field_recall(run, example, **kwargs) -> dict:
    """Calculates what % of expected fields were correctly extracted."""
    predicted = run.outputs.get("extracted_data", {})
    expected = example.outputs.get("expected", {})
    
    if not expected:
        return {"key": "field_recall", "score": 0.0}
    
    # Check each key in the 'Gold' reference
    matches = sum(1 for k, v in expected.items() if predicted.get(k) == v)
    score = matches / len(expected)
    
    return {"key": "field_recall", "score": score}

# --- TARGET FUNCTION ---

def predict(inputs: dict) -> dict:
    """Passes the dataset input into your LangGraph app."""
    # Ensure we use the right key from your dataset (usually 'input')
    raw_text = inputs.get("input") or ""
    result = app.invoke({"raw_text": raw_text, "iterations": 0}, config={"recursion_limit": 10})
    return {"extracted_data": result.get("extracted_data", {})}

# --- MAIN EXECUTION ---

def main():
    print(" Starting Supply Chain Eval...")
    results = client.evaluate(
        predict,
        data="supply_truth_50",
        evaluators=[eval_json_exact, eval_field_recall],
        experiment_prefix="v1-logic-test",
        max_concurrency=5,
    )
    print(" Evaluation Complete. View results in LangSmith.")

if __name__ == "__main__":
    main()