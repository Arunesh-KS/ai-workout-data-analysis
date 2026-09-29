from typing import Literal, Union, List, Annotated
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
    set_index: int = Field(
        description="The exact set number (order) this adjustment applies to."
    )
    new_target_weight_kg: float
    new_target_reps: int
    new_target_rir: int
    ai_instructions: str

# ============================================================
# INVESTIGATION STATE (LIGHTWEIGHT)
# ============================================================

class InvestigationState(BaseModel):
    """
    Ultra-compact memory of the investigation.
    """
    active_hypotheses: List[str] = Field(
        default_factory=list,
        description="List of the exact cause IDs still under investigation."
    )
    ruled_out_hypotheses: List[str] = Field(
        default_factory=list,
        description="List of the exact cause IDs that have been eliminated."
    )
    established_facts: str = Field(
        default="",
        description="CRITICAL MEMORY: Dense bulleted list of exact numbers, weights, and plan details."
    )
    unresolved_questions: str = Field(
        default="",
        description="What specific information is still missing?"
    )

# ============================================================
# COMMANDS
# ============================================================

class CommandQueryDatabase(BaseModel):
    """Request historical workout data for an exercise."""
    action_type: Literal["QUERY_DATABASE"]
    exercise_to_query: str = Field(description="Exact exercise name to query.")
    rationale: str = Field(description="Why this command is needed.")

class CommandAskUser(BaseModel):
    """Ask the user for information."""
    action_type: Literal["ASK_USER"]
    question: str = Field(description="The specific question to ask.")
    rationale: str = Field(description="Why this information is needed.")

class CommandQueryPlan(BaseModel):
    """Request current or previous workout-plan information."""
    action_type: Literal["QUERY_PLAN"]
    query_type: Literal["current", "previous"]
    rationale: str = Field(description="Why plan information is needed.")

class CommandFinalize(BaseModel):
    """Finish the investigation and produce the final prescription."""
    action_type: Literal["FINALIZE_DIAGNOSIS"]
    analysis: str = Field(
        description="Concise final diagnosis explaining the strongest evidence."
    )
    is_override: bool = Field(
        description="Set to true if this overrides the pending_fast_path_updates."
    )
    adjustments: List[ExerciseAdjustment] = Field(
        default_factory=list,
        description="Target adjustments per set. Leave empty if no changes."
    )
    issue_log_updates: List[IssueLogUpdate] = Field(
        description="Always provide an issue-log conclusion."
    )

# ============================================================
# ACTION UNION
# ============================================================

AgentAction = Annotated[
    Union[
        CommandQueryDatabase,
        CommandAskUser,
        CommandQueryPlan,
        CommandFinalize
    ],
    Field(discriminator="action_type")
]

# ============================================================
# MASTER RESPONSE
# ============================================================

class AgentResponse(BaseModel):
    """
    Complete response from the AI.
    """
    # 1. THINK FIRST: The AI must update its whiteboard before issuing commands.
    investigation_state: InvestigationState = Field(
        description="Updated compact state of the investigation."
    )
    
    # 2. ACT SECOND: Issue commands based on the updated state.
    actions: List[AgentAction] = Field(
        description="One or more commands to execute this turn."
    )