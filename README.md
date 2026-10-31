# AgentAudit

AgentAudit is an agentic refund-support application built with LangChain, LangGraph, Groq, and Streamlit.

It demonstrates how an LLM-powered support agent can coordinate tools while relying on deterministic Python logic for refund eligibility and amount calculations.

> **Important:** Refund ticket issuance is simulated. This project does not initiate real payments, bank transfers, or refunds.

## Features

- Conversational refund-support interface built with Streamlit.
- LangChain agent orchestration backed by LangGraph.
- Groq-hosted Qwen model integration.
- Purchase-history lookup against a synthetic dataset of 1,000 users.
- Deterministic refund validation using a 30-day eligibility window.
- Decimal-based refund amount calculation.
- Simulated refund-ticket issuance.
- Tool execution tracing with JSONL records.
- DeepEval-based agent evaluation.

## Architecture

```text
User
 |
 v
Streamlit Interface
 |
 v
LangChain / LangGraph Agent
 |
 +---- Purchase History Tool
 |
 +---- Refund Eligibility and Amount Tool
 |
 +---- Simulated Refund Ticket Tool
 |
 +---- Current Date Tool
 |
 v
Structured Response
 |
 v
User
```

The LLM handles conversation and tool coordination. Python business logic remains authoritative for purchase validation, eligibility, and refund amounts.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Agent orchestration | LangChain / LangGraph |
| Model provider | Groq |
| Configured model | qwen/qwen3.8-27b |
| User interface | Streamlit |
| Structured validation | Pydantic |
| Evaluation | DeepEval |
| Tracing | Custom tool callbacks and JSONL records |

## Project Structure

```text
AgentAudit/
├── agent/
│   ├── .streamlit/
│   │   └── config.toml
│   ├── agent.py
│   ├── data.py
│   ├── models.py
│   ├── objects.py
│   ├── prompts.py
│   ├── tools.py
│   ├── tracing.py
│   ├── ui.py
│   └── users_data_1000.json
├── eval/
│   ├── dataset.py
│   ├── eval_all.py
│   ├── model.py
│   └── individual metric scripts
├── understanding_metrics/
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rj1230/agent_audit.git
cd agent_audit
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If you use `uv`:

```bash
uv pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or expose API credentials.

### 5. Run the application

From the project root:

```bash
streamlit run agent/ui.py
```

Open the local Streamlit URL displayed in the terminal.

## Evaluation

The combined evaluation runner is:

```bash
python -m eval.eval_all
```

It evaluates agent executions using seven metrics:

1. Task Completion
2. Step Efficiency
3. Plan Quality
4. Plan Adherence
5. Argument Correctness
6. Tool Correctness
7. Answer Correctness

The evaluation uses a Groq-hosted judge model. Evaluation consumes additional API tokens and may be affected by provider rate limits.

## Refund Rules

The current refund validator:

- Requires a valid customer username.
- Requires at least one requested item.
- Rejects duplicate item requests.
- Checks that each requested item belongs to the customer.
- Rejects future-dated purchases.
- Enforces a 30-day refund window.
- Calculates the eligible refund amount using Python Decimal arithmetic.
- Rejects the request if any requested item fails validation.

The ticket tool revalidates eligibility before returning a simulated ticket result.

## Tracing

Tool execution traces are written as JSONL records under:

```text
agent/traces/
```

These records support inspection of tool execution, latency, results, and agent behavior.

Trace files may contain customer and conversation data. Treat them as sensitive runtime data and do not commit them.

## Limitations

- Refund tickets are simulated and are not connected to a payment provider.
- The included customer and purchase records are synthetic demonstration data.
- The application does not implement production customer authentication.
- The in-memory conversation checkpointer does not provide durable conversation persistence.
- Evaluation results depend on the configured judge model and provider availability.

## Project Purpose

AgentAudit explores the engineering of tool-using support agents, with emphasis on deterministic business rules, structured responses, tool execution, traceability, and evaluation.

It is a demonstration project, not a production refund-processing system.
