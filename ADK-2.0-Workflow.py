from typing import Any, Dict
from google.adk import Agent, Context, Workflow
from google.adk.workflow import START, node
from pydantic import BaseModel, Field


# 1. Structured Output Schema for [LLM] Analyze History
class HistoryAnalysisVerdict(BaseModel):
    """Typed decision schema emitted by the [LLM] Analyze History node."""

    decision: str = Field(
        ...,
        pattern=r"^(APPROVED|REJECTED)$",
        description="Final upgrade eligibility decision ('APPROVED' or 'REJECTED').",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation grounded in the customer's flight history.",
    )


# 2. Stage 1: [Function] Check Miles (3-Way Conditional Branch)
@node(name="check_miles")
def check_miles_node(ctx: Context, node_input: Any = None) -> Dict[str, Any]:
    """[Function] Check Miles — routes to <5k, 5k-20k, or >20k branches."""
    user_id = ctx.state.get("user_id", "USR-101")
    miles: int = ctx.state.get("miles", 12000)  # Default: 5k < Miles < 20k
    ctx.state["miles"] = miles

    if miles < 5000:
        ctx.route = "MILES_UNDER_5K"
    elif miles > 20000:
        ctx.route = "MILES_OVER_20K"
    else:
        ctx.route = "MILES_5K_TO_20K"

    return {"user_id": user_id, "miles": miles, "route": ctx.route}


# 3. Stage 2a (5k-20k Path): [Human Input] Get Consent
@node(name="get_consent")
def get_consent_node(ctx: Context, node_input: Any = None) -> Dict[str, Any]:
    """[Human Input] Get Consent — captures customer consent before pulling history."""
    consent_granted = bool(ctx.state.get("consent_granted", True))
    ctx.state["consent_granted"] = consent_granted
    return {
        "consent_granted": consent_granted,
        "prompt": "Customer consented to flight history review.",
    }


# 4. Stage 2b (5k-20k Path): [Tool] Fetch Flight History
@node(name="fetch_flight_history")
def fetch_flight_history_node(ctx: Context, node_input: Any = None) -> Dict[str, Any]:
    """[Tool] Fetch Flight History — direct tool node (0 LLM tool-call round-trips)."""
    user_id = ctx.state.get("user_id", "USR-101")
    flight_history = {
        "user_id": user_id,
        "flights_last_12m": 18,
        "cancellations": 2,
        "fare_class": "Y",
        "on_time_payment": True,
    }
    ctx.state["flight_history"] = flight_history
    return flight_history


# 5. Stage 2c (5k-20k Path): [LLM] Analyze History
analyze_history_llm = Agent(
    name="analyze_history",
    model="gemini-3.1-flash-001",
    description="[LLM] Analyze History — evaluates flight history for 5k-20k mile customers.",
    instruction=(
        "You are an Airline Upgrade Eligibility Analyst.\n"
        "Review the customer's flight history in session state (`{flight_history}`).\n"
        "Approve the upgrade ('APPROVED') if the customer has >= 10 flights in the "
        "last 12 months and <= 3 cancellations; otherwise respond with 'REJECTED'."
    ),
    output_schema=HistoryAnalysisVerdict,
)


@node(name="analyze_history_router")
def analyze_history_router_node(
    ctx: Context, node_input: Any = None
) -> Dict[str, Any]:
    """Evaluates the [LLM] Analyze History output and sets ctx.route ('APPROVED' vs 'REJECTED')."""
    verdict_raw = node_input if isinstance(node_input, dict) else {}
    decision = str(
        verdict_raw.get("decision") or ctx.state.get("llm_decision", "APPROVED")
    ).upper()

    if decision == "APPROVED":
        ctx.route = "APPROVED"
    else:
        ctx.route = "REJECTED"

    return {"decision": ctx.route, "llm_output": node_input}


# 6. Convergent Execution & Terminal Nodes
@node(name="process_upgrade")
def process_upgrade_node(ctx: Context, node_input: Any = None) -> Dict[str, Any]:
    """[Function] Process Upgrade — shared by 'Miles > 20k' and LLM 'Approved' paths."""
    user_id = ctx.state.get("user_id", "USR-101")
    ctx.state["upgrade_processed"] = True
    return {
        "user_id": user_id,
        "upgrade_status": "PROCESSED_IN_RESERVATION_SYSTEM",
        "source": node_input,
    }


@node(name="issue_upgrade")
def issue_upgrade_node(ctx: Context, node_input: Any = None) -> Dict[str, Any]:
    """Terminal Node: Issue Upgrade."""
    return {
        "final_outcome": "Issue Upgrade",
        "details": node_input,
    }


@node(name="deny_upgrade")
def deny_upgrade_node(ctx: Context, node_input: Any = None) -> Dict[str, Any]:
    """Terminal Node: Deny Upgrade — shared by 'Miles < 5k' and LLM 'Rejected' paths."""
    return {
        "final_outcome": "Deny Upgrade",
        "details": node_input,
    }


# 7. ADK 2.0 Declarative Workflow Graph (1-to-1 Match with Diagram)
upgrade_workflow_v2 = Workflow(
    name="airline_upgrade_eligibility_workflow_v2",
    description=(
        "ADK 2.0 Graph Workflow matching the attached diagram: "
        "START -> [Function] Check Miles -> (<5k Deny | 5k-20k Consent -> Tool -> LLM | >20k Process Upgrade)."
    ),
    edges=[
        # 3-Way conditional branch out of [Function] Check Miles
        (
            START,
            check_miles_node,
            {
                "MILES_UNDER_5K": deny_upgrade_node,
                "MILES_5K_TO_20K": get_consent_node,
                "MILES_OVER_20K": process_upgrade_node,
            },
        ),
        # Middle branch (5k < Miles < 20k): [Human Input] -> [Tool] -> [LLM]
        (get_consent_node, fetch_flight_history_node),
        (fetch_flight_history_node, analyze_history_llm),
        (analyze_history_llm, analyze_history_router_node),
        # 2-Way conditional branch out of [LLM] Analyze History
        (
            analyze_history_router_node,
            {
                "REJECTED": deny_upgrade_node,
                "APPROVED": process_upgrade_node,
            },
        ),
        # Shared upgrade execution -> Terminal Issue Upgrade
        (process_upgrade_node, issue_upgrade_node),
    ],
)

# Expose `root_agent` for `adk web` / `adk run` / `adk api_server` CLI discovery
root_agent = upgrade_workflow_v2

