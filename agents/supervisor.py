from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from agents.parser_agent import ParserAgent
from agents.critic_agent import CriticAgent
import os
import json

# 1. Define the 'State' (What the agents share)
class AgentState(TypedDict):
    raw_text: str
    extracted_data: dict
    critique: dict
    iterations: int  # We limit loops so it doesn't run forever
    final_output: dict

# 2. Initialize our Agents
parser = ParserAgent()
critic = CriticAgent()

def parser_node(state: AgentState):
    print("--- PARSING DATA ---")
    result = parser.extract_shipment(state["raw_text"])
    # We convert to dict to keep the 'State' serializable
    return {"extracted_data": result.dict(), "iterations": state["iterations"] + 1}

def critic_node(state: AgentState):
    print("--- CRITIQUING DATA ---")
    # Here we would load our mock_database.json
    import json
    with open("mock_database.json", "r") as f:
        db = json.load(f)
    
    from agents.schema.supply_chain import Shipment
    shipment_obj = Shipment(**state["extracted_data"])
    verdict = critic.verify(shipment_obj, db)
    
    return {"critique": verdict.dict()}

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
    The Final Step: Writes the verified 'Truth' to a permanent file.
    """
    print("--- STEP: SAVING TO GOLD VAULT ---")
    
    data = state["extracted_data"]
    shipment_id = data.get("shipment_id", "UNKNOWN_ID")
    
    # Ensure the directory exists
    os.makedirs("data/gold", exist_ok=True)
    
    file_path = f"data/gold/{shipment_id}.json"
    
    # Write the data with 'indent' so it's human-readable
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"--- SUCCESS: Data secured at {file_path} ---")
    return {"final_message": f"Saved to {file_path}"} # This is the final output

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