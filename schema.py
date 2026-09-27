from typing import Literal, Union, List, Optional, Annotated
from pydantic import BaseModel, Field

class IssueLogUpdate(BaseModel):
    date_flagged: str
    exercise: str
    muscle_group: str
    issue_type: str
    description: str
    status: str
    ai_advice: str

class ExerciseAdjustment(BaseModel):
    exercise: str
    new_target_weight_kg: float
    new_target_reps: int
    new_target_rir: int
    ai_instructions: str
   


class ActionQueryDatabase(BaseModel):
    """Triggered when the AI needs historical data for another exercise."""
    action_type: Literal["QUERY_DATABASE"]
    exercise_to_query: str = Field(description="The exact name of the exercise to look up.")
    reasoning: str = Field(description="Internal thought process on why this data is needed.")

class ActionAskUser(BaseModel):
    """Triggered when the AI needs physical symptoms or lifestyle context from the user."""
    action_type: Literal["ASK_USER"]
    question: str = Field(description="The clarifying question to print to the terminal.")
    reasoning: str = Field(description="Internal thought process explaining the current hypothesis.")

class ActionFinalize(BaseModel):
    """Triggered when confidence is high enough to write the final CSV updates."""
    action_type: Literal["FINALIZE_DIAGNOSIS"]
    analysis: str = Field(description="The final biomechanical/fatigue diagnosis.")
    is_override: bool
    adjustments: List[ExerciseAdjustment] = Field(
        default=[], 
        description="List of target changes. Leave empty or omit if no changes are needed."
    )
    issue_log_updates: List[IssueLogUpdate] = Field(description="Always log the conclusion of the investigation here, even if adjustments is empty.")
class ActionQueryPlan(BaseModel):
    """Triggered when the AI needs context about the user's overall routine, current plan structure, or past plans."""
    action_type: Literal["QUERY_PLAN"]
    query_type: Literal["current", "previous"] = Field(description="Whether to fetch the 'current' active plan or the most recent 'previous' plan.")
    reasoning: str = Field(description="Internal thought process on why plan data is needed.")




# --- The Master Agent Schema ---

# The discriminator tells Pydantic to look at the 'action_type' field first, 
# then validate against the corresponding class.
# --- The Master Agent Schema ---

# 1. Rename your existing Union to 'AgentAction'
AgentAction = Annotated[
    Union[ActionQueryDatabase, ActionAskUser, ActionQueryPlan, ActionFinalize], 
    Field(discriminator="action_type")
]

# 2. Create a new root BaseModel that holds a LIST of these actions
class AgentResponse(BaseModel):
    """The root response object from the AI, containing one or more actions."""
    actions: List[AgentAction] = Field(description="A list of actions to execute this turn.")