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
    issue_log_append: Optional[IssueLogUpdate]


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
    adjustments: List[ExerciseAdjustment]

# --- The Master Agent Schema ---

# The discriminator tells Pydantic to look at the 'action_type' field first, 
# then validate against the corresponding class.
AgentResponse = Annotated[
    Union[ActionQueryDatabase, ActionAskUser, ActionFinalize], 
    Field(discriminator="action_type")
]