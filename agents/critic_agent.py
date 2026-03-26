import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from agents.schema.supply_chain import Shipment

load_dotenv()

class ValidationReview(BaseModel):
    is_valid: bool = Field(..., description="True if the data is logically sound.")
    issues: list[str] = Field(default_factory=list, description="List of errors or conflicts found.")
    reconciliation_status: str = Field(..., description="MATCHED, CONFLICT, or UNKNOWN_ENTITY")
    final_decision: str = Field(..., description="A 1-sentence summary of the verdict.")

class CriticAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        self.structured_llm = self.llm.with_structured_output(ValidationReview)

    def verify(self, extracted_data: Shipment, db_context: dict):
        """
        Agent C & D: Compares the 'Parser Output' against 'Mock Database'.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Senior Logistics Auditor. Compare the EXTRACTED DATA against the DATABASE.\n"
                "Rules:\n"
                "1. If SKU is not in 'valid_skus', status is UNKNOWN_ENTITY.\n"
                "2. If Origin is not in 'approved_suppliers', list as a CONFLICT.\n"
                "3. If quantity > 'max_quantity' in PO, list as a CONFLICT.\n"
                "Reject data (is_valid=False) if any of the above occur."
            )),
            ("human", "EXTRACTED DATA: {data}\n\nDATABASE: {db}")
        ])

        chain = prompt | self.structured_llm
        
        return chain.invoke({
            "data": extracted_data.model_dump_json(),
            "db": json.dumps(db_context)
        })
