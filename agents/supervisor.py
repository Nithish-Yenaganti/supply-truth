import operator
import sys
from pathlib import Path
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from datetime import datetime
from pydantic import ValidationError
import re
# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.parser_agent import ParserAgent
from agents.critic_agent import CriticAgent
import os
import json

# 1. Define the 'State' (What the agents share)
class AgentState(TypedDict):
    raw_text: str
    extracted_data: dict
    critique: dict
    iterations: Annotated[int, operator.add]

# 2. Initialize our Agents
parser = ParserAgent()
critic = CriticAgent()

def parser_node(state: AgentState):
    print("--- PARSING DATA ---")
    result = parser.extract_shipment(state["raw_text"])
    if isinstance(result, str):
        return {
            "extracted_data": {},
            "critique": {
                "is_valid": False,
                "issues": [result],
                "reconciliation_status": "PARSER_ERROR",
                "final_decision": "Parser failed before structured extraction.",
            },
            "iterations": state["iterations"] + 1,
        }

    # Keep state JSON-serializable (e.g., datetime -> ISO string).
    return {
        "extracted_data": result.model_dump(mode="json"),
        "iterations": state["iterations"] + 1,
    }

def critic_node(state: AgentState):
    print("--- CRITIQUING DATA ---")
    if not state.get("extracted_data"):
        return {
            "critique": state.get("critique", {
                "is_valid": False,
                "issues": ["Missing extracted_data from parser."],
                "reconciliation_status": "PARSER_ERROR",
                "final_decision": "Parser did not produce structured data.",
            })
        }

    # Load DB relative to project root so cwd does not matter.
    db_path = PROJECT_ROOT / "mock_database.json"
    with open(db_path, "r") as f:
        db = json.load(f)

    # Load previous gold context (if exists) and pass to critic.
    new_data = state["extracted_data"]
    prior_state = {}
    prior_history = []
    shipment_id = new_data.get("shipment_id")
    if shipment_id:
        gold_path = PROJECT_ROOT / "data" / "gold" / f"{shipment_id}.json"
        if gold_path.exists():
            with open(gold_path, "r") as f:
                prior_record = json.load(f)
            if isinstance(prior_record, dict):
                prior_state = prior_record.get("current_state", {}) or {}
                prior_history = prior_record.get("history", []) or []

    from agents.schema.supply_chain import Shipment

    
    try:
        shipment_obj = Shipment(**new_data)
    except ValidationError as e:
        return {"critique": {"is_valid": False,
            "issues": [f"Shipment schema validation failed: {e}"],
            "reconciliation_status": "PARSER_ERROR",
            "final_decision": "Parser returned incomplete or invalid structured data.",
        }
    }

    verdict = critic.verify(
        shipment_obj,
        db,
        prior_state=prior_state,
        history=prior_history,
    )
    
    return {"critique": verdict.model_dump()}

def router(state: AgentState):
    # This is the "Decision" function
    if state["critique"]["is_valid"]:
        return "accept"
    elif state["iterations"] > 3:
        return "fail"
    else:
        return "retry"


def save_to_gold_node(state: AgentState):
    """
    This is your 'Truth Vault' logic. 
    It ensures we keep a history of every update.
    """
    new_data = state["extracted_data"]
    shipment_id = new_data.get("shipment_id", "UNKNOWN")
    os.makedirs("data/gold", exist_ok=True)
    safe_shipment_id = re.sub(r'[^a-zA-Z0-9]', '_', shipment_id)
    file_path = f"data/gold/{safe_shipment_id}.json"

    # 1. Initialize or Load the Record
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            full_record = json.load(f)
    else:
        full_record = {
            "shipment_id": shipment_id,
            "current_state": {},
            "history": []
        }

    # 1b. Migrate/repair legacy record shapes.
    if not isinstance(full_record, dict):
        full_record = {"shipment_id": shipment_id, "current_state": {}, "history": []}
    full_record.setdefault("shipment_id", shipment_id)
    if not isinstance(full_record.get("current_state"), dict):
        legacy_state = {
            k: v
            for k, v in full_record.items()
            if k not in {"shipment_id", "current_state", "history"}
        }
        full_record["current_state"] = legacy_state
    if not isinstance(full_record.get("history"), list):
        full_record["history"] = []

    # Skip log bloat: if nothing changed, do not create a new history version.
    if full_record["current_state"] == new_data:
        print(f"Record {shipment_id} unchanged; no new history version.")
        return state

    # 2. Add the update to History
    next_version = max(
        (
            entry.get("version", 0)
            for entry in full_record["history"]
            if isinstance(entry, dict) and isinstance(entry.get("version"), int)
        ),
        default=0,
    ) + 1
    history_entry = {
        "version": next_version,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "changes": new_data
    }
    full_record["history"].append(history_entry)

    # 3. Update the Current State (The 'Latest Truth')
    # Replace, don't merge, so current_state always mirrors latest extraction exactly.
    full_record["current_state"] = dict(new_data)

    # 4. Final Save (atomic write)
    temp_path = f"{file_path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(full_record, f, indent=4)
    os.replace(temp_path, file_path)
     
    print(f"Record {shipment_id} updated to Version {len(full_record['history'])}")
    return state

# 4. Build the Graph

# --- 1. Initialize the Graph ---
workflow = StateGraph(AgentState)

# --- 2. Add ALL Nodes (The Workers) ---
workflow.add_node("parser", parser_node)           # Agent A & B
workflow.add_node("critic", critic_node)           # Agent C & D
workflow.add_node("save_to_gold", save_to_gold_node) # The Vault (Step 5)

# --- 3. Define the START and Fixed Edges ---
workflow.set_entry_point("parser")
workflow.add_edge("parser", "critic")

# --- 4. Define the Conditional Logic (The Router) ---
workflow.add_conditional_edges(
    "critic", 
    router, 
    {
        "accept": "save_to_gold",  # If clean, go to the Vault
        "retry": "parser",        # If dirty, try again
        "fail": END               # If hopeless, stop
    }
)

# --- 5. Define the Final Exit ---
# Once saved to gold, the process is finished.
workflow.add_edge("save_to_gold", END)

# --- 6. Compile the Application ---
app = workflow.compile()


if __name__ == "__main__":
    # The messy input that needs cleaning
    initial_input = {
        "raw_text": """
        Shipment ID: MERC-550
        Origin: Singapore Global Port
        Destination: Los Angeles
        ETA: 2026-04-10
        Items: 200 x HDC-09
        """,
        
        "iterations": 0,
    }

    print("Starting SupplyChain Truth Engine...\n")
    
    # Run the graph until it hits 'END'
    final_state = app.invoke(initial_input)

    print("\n--- FINAL PROCESS SUMMARY ---")
    if final_state["critique"]["is_valid"]:
        print("RESULT: Successfully Verified and Saved.")
        print(f"FINAL DATA: {final_state['extracted_data']}")
    else:
        print(f"RESULT: Failed. Human intervention required.")
        print(f"REASON: {final_state['critique']['final_decision']}")
