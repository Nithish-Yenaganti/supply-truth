from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from agents.parser_agent import ParserAgent
from agents.critic_agent import CriticAgent

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
    
    from schema.supply_chain import Shipment
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

# 3. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("parser", parser_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "critic")

workflow.add_conditional_edges(
    "critic",
    router,
    {
        "accept": END,
        "retry": "parser", # GO BACK AND FIX IT
        "fail": END
    }
)

app = workflow.compile()