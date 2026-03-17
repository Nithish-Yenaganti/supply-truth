from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

class LineItem(BaseModel):
    sku: str = Field(..., description="Unique product identifier (e.g., APPL-101)")
    quantity: int = Field(..., gt=0, description="Number of units, must be a positive integer")
    description: Optional[str] = Field(None, description="Short text describing the item")

class Shipment(BaseModel):
    shipment_id: str = Field(..., description="Unique ID for the shipment (e.g., MERC-992)")
    origin: str = Field(..., description="City or Port of origin")
    destination: str = Field(..., description="Final destination city/port")
    eta: datetime = Field(..., description="Estimated Time of Arrival in ISO format")
    items: List[LineItem] = Field(..., description="List of products in this shipment")
    
    # This is a 'Constraint' - Agent C will use this logic
    @field_validator('eta')
    @classmethod
    def eta_must_be_plausible(cls, v):
        # If the year is way off, it's a hallucination
        if v.year < 2024 or v.year > 2030:
            raise ValueError("ETA Year is physically implausible. Check data source.")
        return v

class ExtractionResult(BaseModel):
    """The final wrapper for our 'Gold' data"""
    data: Optional[Shipment] = None
    confidence_score: float = Field(..., ge=0, le=1)
    reasoning: str = Field(..., description="The Agent's explanation for its decisions")