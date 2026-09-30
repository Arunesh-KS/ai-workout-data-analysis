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
class CommandRemoveHistory(BaseModel):
    """Prune raw exercise history from the active context once it's cleared."""
    action_type: Literal["REMOVE_HISTORY"]
    exercise: str = Field(description="The exercise history to strip away.")
    replace_note: str = Field(description="A 1-sentence static note confirming the status of this exercise.")

class CommandRemovePreviousPlan(BaseModel):
    """Prune legacy plan details once the program comparison is complete."""
    action_type: Literal["REMOVE_PREVIOUS_PLAN"]
    replace_note: str = Field(description="A concise note summarizing the volume delta (e.g., 'Old plan had 2 sets, new plan has 1 set').")

# ============================================================
# ACTION UNION
# ============================================================

AgentAction = Annotated[
    Union[
        CommandQueryDatabase,
        CommandAskUser,
        CommandQueryPlan,
        CommandFinalize,
        CommandRemoveHistory,
        CommandRemovePreviousPlan
    ],
    Field(discriminator="action_type")
]

# ============================================================
# MASTER RESPONSE (MESSAGE APPEND MODE)
# ============================================================

class AgentResponse(BaseModel):
    """
    Complete response from the AI containing only actions.
    """
    actions: List[AgentAction] = Field(
        description="One or more commands to execute this turn."
    )