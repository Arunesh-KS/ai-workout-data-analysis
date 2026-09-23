from pydantic import BaseModel, Field
from typing import Optional

class AIWorkoutAnalysis(BaseModel):
    is_override: bool = Field(
        description="Set to True ONLY IF the user is stalling, reports pain/injury, or requires a variation. Set to False if the notes are positive or no changes to the default progression are needed."
    )
    
    # Override fields (Only used if is_override is True)
    override_weight_kg: Optional[float] = Field(
        default=None,
        description="The new suggested weight in kg. Use this if recommending a deload or a different exercise variation."
    )
    override_reps: Optional[int] = Field(
        default=None,
        description="The new suggested rep count (e.g., around 12-15 for joint recovery or  around 5-8 for strength blocks . also give the starting rep )."
    )
    suggested_variation: Optional[str] = Field(
        default=None,
        description="A suggested alternative exercise if the current one causes pain or extreme stalling (e.g., 'Switch to Dumbbell Bench Press' if barbell hurts elbows)."
    )
    
    # Always provided to give the user context
    coach_feedback: str = Field(
        description="Actionable advice addressing the specific notes or stalling. if weights , reps were adjusted , also provide the rep range , give cues for form ."
    )
