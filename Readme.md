# adk2-0-vs-langgraph

Comparison of a graph workflow definition between **Google ADK 2.0** and **LangGraph** using a real-world Airline Upgrade Eligibility example.

## Workflow Overview

The diagram below illustrates the end-to-end Airline Upgrade Eligibility decision flow implemented in both [ADK-2.0-Workflow.py](file:///Users/syedjaffry/src/adk2-0-vs-langgraph/ADK-2.0-Workflow.py) and [LangGraph-Workflow.py](file:///Users/syedjaffry/src/adk2-0-vs-langgraph/LangGraph-Workflow.py):

![Airline Upgrade Eligibility Workflow Design](workflow-design-adk2.svg)

### Workflow Description & Logic

The workflow models a tiered decision process that evaluates whether a customer is eligible for a flight upgrade based on their loyalty mileage and flight history:

1. **Start & Mileage Evaluation (`[Function] Check Miles`)**:
   - The workflow begins at `START` and executes the `Check Miles` function node.
   - It performs a **3-way conditional branch** based on the customer's accumulated miles:
     - **`< 5,000 miles` (Under 5k)**: Automatically routes directly to the terminal **Deny Upgrade** node.
     - **`> 20,000 miles` (Over 20k)**: Fast-tracks directly to **Process Upgrade** and terminates at **Issue Upgrade**.
     - **`5,000 – 20,000 miles` (5k–20k Review Tier)**: Transitions into the multi-stage evaluation pipeline requiring consent, data retrieval, and LLM reasoning.

2. **Mid-Tier Evaluation Pipeline (5,000 – 20,000 miles)**:
   - **`[Human Input] Get Consent`**: Prompts the customer for permission to inspect their past 12-month flight history (demonstrating human-in-the-loop interaction and interrupt handling).
   - **`[Tool] Fetch Flight History`**: Once consent is received, directly invokes a data fetching tool to pull flight records (cancellations, fare class, completed flights) without incurring an LLM tool-calling round trip.
   - **`[LLM] Analyze History`**: Uses Gemini (`gemini-3.1-flash-001`) with structured output (`HistoryAnalysisVerdict`) to evaluate whether the customer meets upgrade thresholds (e.g., $\ge 10$ flights and $\le 3$ cancellations in the past 12 months).

3. **Convergence & Terminal Outcomes**:
   - **Approval Path**: If the LLM returns `APPROVED`, it converges with the high-mileage path into **`[Function] Process Upgrade`**, which executes the reservation system update and concludes at **`Issue Upgrade`**.
   - **Rejection Path**: If the LLM returns `REJECTED`, it converges with the low-mileage path and concludes at **`Deny Upgrade`**.

---

## Architectural Comparison

| Feature | Google ADK 2.0 ([`ADK-2.0-Workflow.py`](file:///Users/syedjaffry/src/adk2-0-vs-langgraph/ADK-2.0-Workflow.py)) | LangGraph ([`LangGraph-Workflow.py`](file:///Users/syedjaffry/src/adk2-0-vs-langgraph/LangGraph-Workflow.py)) |
| :--- | :--- | :--- |
| **Workflow Definition** | Declarative `Workflow(edges=[...])` list linking nodes and routing dictionaries | Graph builder pattern (`StateGraph`) with explicit `add_node`, `add_edge`, and `add_conditional_edges` |
| **State Management** | Implicit shared session state managed via `ctx.state` | Explicit schema enforced via a typed dictionary (`TypedDict`, e.g., `UpgradeState`) |
| **Branching / Routing** | In-node routing by setting `ctx.route = "ROUTE_NAME"` or dictionary-based conditional edges | Dedicated routing functions evaluated on state return values passed to `add_conditional_edges` |
| **LLM Integration** | First-class `Agent` node placed directly into the edge graph (`(fetch_flight_history, analyze_history_llm)`) | LLM invocation wrapped inside a Python node function invoking `ChatGoogleGenerativeAI` |
| **Human-in-the-Loop** | Framework-managed pause/resume points integrated with ADK context | Graph pauses with `interrupt(...)` requiring a checkpointer (`MemorySaver`) to resume |

---

## Files in this Repository

- [workflow-design-adk2.svg](file:///Users/syedjaffry/src/adk2-0-vs-langgraph/workflow-design-adk2.svg): Visual workflow architecture diagram.
- [ADK-2.0-Workflow.py](file:///Users/syedjaffry/src/adk2-0-vs-langgraph/ADK-2.0-Workflow.py): Implementation using Google ADK 2.0 Workflow and Agent APIs.
- [LangGraph-Workflow.py](file:///Users/syedjaffry/src/adk2-0-vs-langgraph/LangGraph-Workflow.py): Equivalent implementation using LangGraph `StateGraph`, `interrupt`, and `MemorySaver`.
