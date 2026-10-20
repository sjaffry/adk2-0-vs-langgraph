from typing import Any, Dict, Literal, Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field


# 1. Explicit Shared Graph State Schema (Required by LangGraph StateGraph)
class UpgradeState(TypedDict, total=False):
    user_id: str
    miles: int
    consent_granted: bool
    flight_history: Dict[str, Any]
    llm_decision: Literal["APPROVED", "REJECTED"]
    llm_reasoning: str
    upgrade_processed: bool
    final_outcome: str


# 2. Structured Output Schema for [LLM] Analyze History
class HistoryAnalysisVerdict(BaseModel):
    """Typed decision schema emitted by the [LLM] Analyze History node."""

    decision: Literal["APPROVED", "REJECTED"] = Field(
        ...,
        description="Final upgrade eligibility decision ('APPROVED' or 'REJECTED').",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation grounded in the customer's flight history.",
    )


# 3. Stage 1: [Function] Check Miles + Conditional Edge Router
def check_miles_node(state: UpgradeState) -> Dict[str, Any]:
    """[Function] Check Miles — populates customer mileage in graph state."""
    user_id = state.get("user_id", "USR-101")
    miles = state.get("miles", 12000)  # Default: 5k < Miles < 20k
    return {"user_id": user_id, "miles": miles}


def route_by_miles(
    state: UpgradeState,
) -> Literal["MILES_UNDER_5K", "MILES_5K_TO_20K", "MILES_OVER_20K"]:
    """LangGraph conditional edge function (replaces ADK 2.0 `ctx.route`)."""
    miles = state.get("miles", 0)
    if miles < 5000:
        return "MILES_UNDER_5K"
    elif miles > 20000:
        return "MILES_OVER_20K"
    return "MILES_5K_TO_20K"


# 4. Stage 2a (5k-20k Path): [Human Input] Get Consent
def get_consent_node(state: UpgradeState) -> Dict[str, Any]:
    """[Human Input] Get Consent — pauses via LangGraph `interrupt()` if consent is missing."""
    consent: Optional[bool] = state.get("consent_granted")
    if consent is None:
        # Pauses graph execution until human resumes with Command(resume=True)
        consent = bool(
            interrupt("Do you consent to a review of your 12-month flight history?")
        )
    return {"consent_granted": consent}


# 5. Stage 2b (5k-20k Path): [Tool] Fetch Flight History
def fetch_flight_history_node(state: UpgradeState) -> Dict[str, Any]:
    """[Tool] Fetch Flight History — direct Python node (0 LLM tool-call round-trips)."""
    user_id = state.get("user_id", "USR-101")
    flight_history = {
        "user_id": user_id,
        "flights_last_12m": 18,
        "cancellations": 2,
        "fare_class": "Y",
        "on_time_payment": True,
    }
    return {"flight_history": flight_history}


# 6. Stage 2c (5k-20k Path): [LLM] Analyze History + Conditional Edge Router
def analyze_history_llm_node(state: UpgradeState) -> Dict[str, Any]:
    """[LLM] Analyze History — wraps ChatGoogleGenerativeAI inside a LangGraph node function."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-001", temperature=0.0)
    structured_llm = llm.with_structured_output(HistoryAnalysisVerdict)

    prompt = (
        "You are an Airline Upgrade Eligibility Analyst.\n"
        f"Review the customer's flight history: {state.get('flight_history')}.\n"
        "Approve the upgrade ('APPROVED') if the customer has >= 10 flights in the "
        "last 12 months and <= 3 cancellations; otherwise respond with 'REJECTED'."
    )
    verdict: HistoryAnalysisVerdict = structured_llm.invoke(prompt)  # type: ignore[assignment]
    return {
        "llm_decision": verdict.decision,
        "llm_reasoning": verdict.reasoning,
    }


def route_by_llm_decision(
    state: UpgradeState,
) -> Literal["APPROVED", "REJECTED"]:
    """LangGraph conditional edge function after [LLM] Analyze History."""
    if state.get("llm_decision") == "APPROVED":
        return "APPROVED"
    return "REJECTED"


# 7. Convergent Execution & Terminal Nodes
def process_upgrade_node(state: UpgradeState) -> Dict[str, Any]:
    """[Function] Process Upgrade — shared by 'Miles > 20k' and LLM 'Approved' paths."""
    return {"upgrade_processed": True}


def issue_upgrade_node(state: UpgradeState) -> Dict[str, Any]:
    """Terminal Node: Issue Upgrade."""
    return {"final_outcome": "Issue Upgrade"}


def deny_upgrade_node(state: UpgradeState) -> Dict[str, Any]:
    """Terminal Node: Deny Upgrade — shared by 'Miles < 5k' and LLM 'Rejected' paths."""
    return {"final_outcome": "Deny Upgrade"}


# 8. Build & Compile the LangGraph StateGraph
builder = StateGraph(UpgradeState)

# Register all 7 nodes explicitly
builder.add_node("check_miles", check_miles_node)
builder.add_node("get_consent", get_consent_node)
builder.add_node("fetch_flight_history", fetch_flight_history_node)
builder.add_node("analyze_history_llm", analyze_history_llm_node)
builder.add_node("process_upgrade", process_upgrade_node)
builder.add_node("issue_upgrade", issue_upgrade_node)
builder.add_node("deny_upgrade", deny_upgrade_node)

# Entry edge + 3-Way conditional branch out of [Function] Check Miles
builder.add_edge(START, "check_miles")
builder.add_conditional_edges(
    "check_miles",
    route_by_miles,
    {
        "MILES_UNDER_5K": "deny_upgrade",
        "MILES_5K_TO_20K": "get_consent",
        "MILES_OVER_20K": "process_upgrade",
    },
)

# Middle branch (5k < Miles < 20k): [Human Input] -> [Tool] -> [LLM]
builder.add_edge("get_consent", "fetch_flight_history")
builder.add_edge("fetch_flight_history", "analyze_history_llm")

# 2-Way conditional branch out of [LLM] Analyze History
builder.add_conditional_edges(
    "analyze_history_llm",
    route_by_llm_decision,
    {
        "REJECTED": "deny_upgrade",
        "APPROVED": "process_upgrade",
    },
)

# Shared upgrade execution -> Terminal Issue Upgrade -> END
builder.add_edge("process_upgrade", "issue_upgrade")
builder.add_edge("issue_upgrade", END)
builder.add_edge("deny_upgrade", END)

# Compile with MemorySaver checkpointer (required for interrupt() human-in-the-loop)
upgrade_workflow_langgraph = builder.compile(checkpointer=MemorySaver())
