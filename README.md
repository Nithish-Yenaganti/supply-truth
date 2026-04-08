# SupplyTruth: Multi-Agent Logistics Extraction and Validation

SupplyTruth is a production-grade agentic system designed to extract, validate, and reconcile shipment data from unstructured logistics communications. Built with LangGraph and Gemini 3.1, the system employs a dual-agent architecture to ensure high-fidelity data capture for supply chain operations.

---

## System Architecture

The project utilizes a supervisor-managed graph consisting of two specialized agents:

1. Parser Agent: Extracts structured shipping data (SKUs, quantities, dates, and carrier codes) from raw email text or documents.
2. Critic Agent: Validates the extracted data against business logic and identifies discrepancies or missing information.

The agents operate in a feedback loop. If the Critic identifies issues, the state is passed back to the Parser for correction. This process continues until the Critic approves the output or a maximum iteration limit is reached.

---

## Technical Stack

* Orchestration: LangGraph
* Models: Gemini 3.1 Pro and 3.1 Flash-Lite
* Inference Engine: GMI Cloud (H100/H200 Infrastructure)
* Observability: LangSmith
* Caching: SQLite and In-Memory
* Language: Python 3.11+

---

## Project Structure

SUPPLYTRUTH/
├── agents/
│   ├── schema/
│   │   └── supply_chain.py      # Pydantic models and AgentState
│   ├── parser_agent.py          # Logic for data extraction
│   ├── critic_agent.py          # Logic for validation and critique
│   └── supervisor.py            # LangGraph workflow definition
├── evals/
│   ├── data_set.jsonl           # 50-sample evaluation dataset
│   └── run_evals.py             # LangSmith evaluation script
├── .env                         # API keys and configuration
└── requirements.txt             # Project dependencies

---

## Key Features

### Agentic State Management
The system uses a centralized AgentState to track raw input, extracted JSON, critique history, and iteration counts. It employs reducers for the iteration counter to maintain state integrity across nodes.

### Observability and Evaluation
Integrated with LangSmith to provide:
* Full trace visibility for every agent decision.
* Automated evaluation runs for accuracy and latency.
* Custom evaluators for exact-match and partial-match scoring.

### High-Performance Inference
Configured to use GMI Cloud endpoints to minimize latency, particularly for the Flash-Lite model used in the high-volume parsing node.

---

## Setup and Installation

### 1. Environment Configuration
Create a .env file in the root directory with the following variables:

GMI_CLOUD_API_KEY=your_api_key
BASE_URL=https://api.gmi-serving.com/v1
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key

### 2. Dependency Installation
pip install -r requirements.txt

### 3. Running Evaluations
To execute the 50-sample evaluation suite:
python -m evals.run_evals

---

## Development Roadmap

* Persistence: Implementation of SQLite Checkpointers for thread-safe session memory.
* Knowledge Retrieval: Integration with Pinecone Vector Database for RAG-based SOP lookups.
* Human-in-the-Loop: Integration of breakpoints for manual approval of high-value shipment discrepancies.
* Deployment: Containerization via Docker for GCP Cloud Run or AWS Lambda deployment.