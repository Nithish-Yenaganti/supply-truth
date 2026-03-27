import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    def _is_missing(self, value):
        return value in (None, "", [], {})

    def _detect_regressions(self, new_data: dict, prior_state: dict) -> list[str]:
        regressions: list[str] = []
        if not isinstance(prior_state, dict) or not prior_state:
            return regressions

        for key in ("origin", "destination", "eta"):
            if not self._is_missing(prior_state.get(key)) and self._is_missing(new_data.get(key)):
                regressions.append(key)

        prior_items = prior_state.get("items") or []
        new_items = new_data.get("items") or []
        if prior_items:
            if not new_items:
                regressions.append("items")
            else:
                prior_item0 = prior_items[0] if isinstance(prior_items[0], dict) else {}
                new_item0 = new_items[0] if isinstance(new_items[0], dict) else {}
                for key in ("sku", "quantity"):
                    if not self._is_missing(prior_item0.get(key)) and self._is_missing(new_item0.get(key)):
                        regressions.append(f"items[0].{key}")

        return regressions

    def verify(
        self,
        extracted_data: Shipment,
        db_context: dict,
        prior_state: dict | None = None,
        history: list | None = None,
    ):
        """
        Agent C & D: Compares the parser output against prior vault state and DB policy.
        """
        # Regression guard: if latest extraction dropped previously known fields, reject early.
        regressions = self._detect_regressions(
            extracted_data.model_dump(mode="json"),
            prior_state or {},
        )
        if regressions:
            return ValidationReview(
                is_valid=False,
                issues=[
                    "Field regression detected vs previous version: "
                    + ", ".join(regressions)
                ],
                reconciliation_status="CONFLICT",
                final_decision="Latest extraction dropped previously known fields.",
            )

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


if __name__ == "__main__":
    critic = CriticAgent()

    mock_db_path = Path(__file__).resolve().parents[1] / "mock_database.json"
    with open(mock_db_path, "r") as f:
        mock_db = json.load(f)

    sample_shipment = Shipment(
        shipment_id="MERC-550",
        origin="Singapore Global Port",
        destination="Los Angeles",
        eta="2026-04-10T00:00:00Z",
        items=[
            {
                "sku": "HDC-09",
                "quantity": 200,
                "description": "Heavy-Duty Crates",
            }
        ],
    )

    try:
        verdict = critic.verify(sample_shipment, mock_db)
        print("--- CRITIC VERDICT ---")
        print(verdict.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error during critic test: {e}")
