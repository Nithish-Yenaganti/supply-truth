import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schema.supply_chain import Shipment  # Our 'Truth' from Step 1

# Load environment variables (API Keys)
load_dotenv()

class ParserAgent:
    def __init__(self):
        # We use Gemini 1.5 Pro for its reasoning depth
        # temperature=0 ensures we get facts, not 'creative' hallucinations
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
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

