import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.schema.supply_chain import Shipment

# Load environment variables (API Keys)
load_dotenv()

class ParserAgent:
    def __init__(self):
        # We use Gemini 1.5 Pro for its reasoning depth
        # temperature=0 ensures we get facts, not 'creative' hallucinations
        self.llm = ChatOpenAI(
            model="google/gemini-3.1-flash-lite-preview",
            temperature=0,
            api_key=os.getenv("GMI_CLOUD_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            max_tokens=10000,

        )
        
        # This binds our schema to the LLM permanently for this agent
        self.structured_llm = self.llm.with_structured_output(Shipment)

    def extract_shipment(self, raw_input: str):
        """
        Agent A reads the text, Agent B maps it to the schema.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert Supply Chain Auditor. "
                "Your task is to extract shipment data from messy unstructured text. "
                "Follow these rules strictly:\n"
                "1. If a value is missing, return null.\n"
                "2. Ensure dates are in ISO format (YYYY-MM-DD).\n"
                "3. Do not invent data; only extract what is present."
            )),
            ("human", "{text}")
        ])

        # Create the chain
        chain = prompt | self.structured_llm
        
        try:
            # Execution
            result = chain.invoke({"text": raw_input})
            return result
        except Exception as e:
            return f"Error during parsing: {str(e)}"



if __name__ == "__main__":
    parser = ParserAgent()
    sample_text = (
        "Shipment ID: MERC-550. Origin: Singapore Global Port. "
        "Destination: Los Angeles. ETA: 2026-04-10. Items: 200 x HDC-09."
    )
   
    print("Running------")
    result = parser.extract_shipment(sample_text)
    if isinstance(result, str):
        print(result)
    else:
        print(result.model_dump())
