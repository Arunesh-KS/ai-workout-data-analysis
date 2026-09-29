from typing import Literal, Union, List, Optional, Annotated
from pydantic import BaseModel, Field


# ============================================================
# ISSUE LOG
# ============================================================

class IssueLogUpdate(BaseModel):
    date_flagged: str
    exercise: str
    muscle_group: str
    issue_type: str
    description: str
    status: str
    ai_advice: str


# ============================================================
# EXERCISE ADJUSTMENT
# ============================================================

class ExerciseAdjustment(BaseModel):
    exercise: str
    new_target_weight_kg: float
    new_target_reps: int
    new_target_rir: int
    ai_instructions: str


# ============================================================
# INVESTIGATION STATE
# ============================================================

CauseStatus = Literal[
    "unknown",
    "possible",
    "supported",
    "strongly_supported",
    "weakened",
    "ruled_out"
]

CauseId = Literal[
    "inadequate_recovery", "inadequate_stimulus", 
    "training_load_mismatch", "pain_soreness", 
    "technique_issue", "program_change", 
    "exercise_order", "local_interference", 
    "target_mismatch", "measurement_issue"
]
class PotentialCause(BaseModel):
    """
    Current status of one possible explanation for the performance problem.

    Evidence lists should contain only concise, relevant evidence.
    Do not provide supporting_evidence for ruled-out causes.
    """
    cause_id: CauseId = Field(
        description="Stable identifier for the cause."
    )

    status: CauseStatus = Field(
        description="Current evidence status of this possible cause."
    )

    supporting_evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Concise evidence supporting this cause. "
            "Leave empty when the cause is ruled out."
        )
    )

    evidence_against: List[str] = Field(
        default_factory=list,
        description=(
            "Concise evidence weakening or contradicting this cause."
        )
    )


class InvestigationState(BaseModel):
    """
    Compact memory of the investigation.
    This is carried between API calls instead of replaying the entire
    conversation history.
    """

    potential_causes: List[PotentialCause] = Field(
        description="Current status of the predefined and any newly discovered causes."
    )

    established_facts: List[str] = Field(
        default_factory=list,
        description=(
            "Short, objective facts established during the investigation."
        )
    )

    active_hypotheses: List[str] = Field(
        default_factory=list,
        description=(
            "Current working explanations that remain under investigation."
        )
    )

    unresolved_questions: List[str] = Field(
        default_factory=list,
        description=(
            "Important questions whose answers could materially change the diagnosis."
        )
    )


# ============================================================
# ACTIONS
# ============================================================

class ActionQueryDatabase(BaseModel):
    """Request historical workout data for an exercise."""
    action_type: Literal["QUERY_DATABASE"]

    exercise_to_query: str = Field(
        description="Exact exercise name to query."
    )

    rationale: str = Field(
        description=(
            "Brief explanation of why this query is useful for "
            "testing the current hypotheses."
        )
    )


class ActionAskUser(BaseModel):
    """Ask the user for information that cannot be retrieved from the database."""
    action_type: Literal["ASK_USER"]

    question: str = Field(
        description="The specific question to ask the user."
    )

    rationale: str = Field(
        description=(
            "Brief explanation of why this information is needed "
            "to distinguish between remaining hypotheses."
        )
    )


class ActionQueryPlan(BaseModel):
    """Request current or previous workout-plan information."""
    action_type: Literal["QUERY_PLAN"]

    query_type: Literal["current", "previous"] = Field(
        description=(
            "Whether to retrieve the current active plan "
            "or the most recent previous plan."
        )
    )

    rationale: str = Field(
        description=(
            "Brief explanation of why plan information is needed."
        )
    )


class ActionFinalize(BaseModel):
    """Finish the investigation and produce the final prescription."""
    action_type: Literal["FINALIZE_DIAGNOSIS"]

    analysis: str = Field(
        description=(
            "Concise final diagnosis explaining the strongest evidence "
            "and why the selected conclusion is justified."
        )
    )

    is_override: bool = Field(
        description=(
            "Whether the normal deterministic progression target "
            "should be overridden."
        )
    )

    adjustments: List[ExerciseAdjustment] = Field(
        default_factory=list,
        description=(
            "Target adjustments. Empty when no program change is justified."
        )
    )

    issue_log_updates: List[IssueLogUpdate] = Field(
        description=(
            "Always provide an issue-log conclusion, even when no "
            "adjustment is required."
        )
    )


# ============================================================
# ACTION UNION
# ============================================================

AgentAction = Annotated[
    Union[
        ActionQueryDatabase,
        ActionAskUser,
        ActionQueryPlan,
        ActionFinalize
    ],
    Field(discriminator="action_type")
]


# ============================================================
# MASTER RESPONSE
# ============================================================

class AgentResponse(BaseModel):
    """
    Complete response from the AI.

    actions:
        What the backend/user should do now.

    investigation_state:
        Compact memory of what has been established so far.
        The backend can carry this state into the next API call.
    """

    actions: List[AgentAction] = Field(
        description=(
            "One or more actions to execute this turn. "
            "Independent queries may be batched."
        )
    )

    investigation_state: InvestigationState = Field(
        description=(
            "Updated compact state of the investigation after considering "
            "all evidence available in this turn."
        )
    )