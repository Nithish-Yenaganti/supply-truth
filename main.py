import json
from agents.parser_agent import ParserAgent # Adjust based on your actual class name

# 1. Pick one case from your jsonl
sample_input = "PO #99102: Please send 500 units of NEEM-BRUSH-01 to our Austin warehouse by Friday."

# 2. Run ONLY the parser
parser = ParserAgent()
result = parser.extract_shipment({"raw_text": sample_input})

# 3. PRINT EVERYTHING
print("--- RAW PARSER OUTPUT ---")
print(result.model_dump())
