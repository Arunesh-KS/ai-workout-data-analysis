import os
import json
import time
from engine import WorkoutEngine
from schema import AIWorkoutAnalysis

# ==========================================
# API CLIENT INITIALIZATION
# ==========================================

# --- GROQ (Active) ---
from groq import Groq
client = Groq()

# --- GEMINI (Commented Out) ---
# from google import genai
# client = genai.Client()

# ==========================================

def needs_ai_intervention(notes, history):
    """The Router: Triggers the Slow Path for notes OR performance regressions."""
    has_notes = isinstance(notes, str) and notes.strip() != ""
    if has_notes:
        return True, "User provided notes."
        
    if len(history) >= 2:
        last_session = history.iloc[-1]
        prev_session = history.iloc[-2]
        
        last_weight = max(last_session['Weight (kg)'])
        prev_weight = max(prev_session['Weight (kg)'])
        
        last_reps = max([r for w, r in zip(last_session['Weight (kg)'], last_session['Reps']) if w == last_weight])
        prev_reps = max([r for w, r in zip(prev_session['Weight (kg)'], prev_session['Reps']) if w == prev_weight])
        
        if last_weight < prev_weight:
            return True, f"Regression: Weight dropped from {prev_weight}kg to {last_weight}kg."
            
        if last_weight == prev_weight and last_reps < prev_reps:
            return True, f"Regression: Reps dropped from {prev_reps} to {last_reps} at {last_weight}kg."
            
        if len(history) >= 3:
            prev2_session = history.iloc[-3]
            prev2_weight = max(prev2_session['Weight (kg)'])
            prev2_reps = max([r for w, r in zip(prev2_session['Weight (kg)'], prev2_session['Reps']) if w == prev2_weight])
            
            if last_weight <= prev2_weight and last_reps <= prev2_reps:
                return True, f"Stalling: No net progress over 3 sessions."

    return False, "Clear path. Progressing normally."

def get_ai_coaching(exercise_name, history, default_target, notes, trigger_reason):
    """The Slow Path: Calls the LLM to get an override decision."""
    
    recent_history = history.tail(3).to_dict(orient='records')
    
    prompt = f"""
    You are an elite biomechanics and strength coach. 
    The user is performing: {exercise_name}
    
    The deterministic system routed this exercise to you because: {trigger_reason}
    
    Recent History (Last 3 sessions):
    {json.dumps(recent_history, indent=2, default=str)}
    
    The deterministic algorithm suggests this default next target:
    {json.dumps(default_target, indent=2, default=str)}
    
    User's Notes from the last session: "{notes}"
    
    INSTRUCTIONS:
    1. Read the trigger reason and the user's notes.
    2. If the user is stalling, regressing, or reporting pain/form breakdown, set `is_override` to True and provide a new target weight/reps (e.g., a deload) or a `suggested_variation`.
    3. If the notes are strictly positive and progress is fine, set `is_override` to False, let the algorithm's target stand, and offer brief encouragement in `coach_feedback`.
    4. Provide specific biomechanical cues or progression advice in `coach_feedback` addressing the stall or form issues.
    5. look for advice based on the user's current strength level , irrespetive of generic advice .

    
    OUTPUT FORMAT:
    You MUST output valid JSON matching this schema. Do not include markdown code blocks, just raw JSON:
    {AIWorkoutAnalysis.model_json_schema()}
    """
    
    # ==========================================
    # GROQ IMPLEMENTATION (Active)
    # ==========================================
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a professional strength and conditioning AI. You output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7  
    )
    
    raw_json_string = response.choices[0].message.content
    parsed_response = AIWorkoutAnalysis.model_validate_json(raw_json_string)
    return parsed_response

    # ==========================================
    # GEMINI IMPLEMENTATION (Commented Out)
    # ==========================================
    # max_retries = 3
    # for attempt in range(max_retries):
    #     try:
    #         response = client.models.generate_content(
    #             model='gemini-3.5-flash-lite',
    #             contents=prompt,
    #             config={
    #                 'response_mime_type': 'application/json',
    #                 'response_schema': AIWorkoutAnalysis,
    #                 'temperature': 0.2, # Change to 0.7-0.9 for a more conversational coaching tone
    #                 'thinking_config': {'thinking_budget': 0}
    #             }
    #         )
    #         return response.parsed
    #         
    #     except Exception as e:
    #         if attempt < max_retries - 1:
    #             print(f"      [Server busy... retrying in 5 seconds (Attempt {attempt + 2}/{max_retries})]")
    #             time.sleep(5)
    #         else:
    #             raise e

def run_pipeline():
    csv_path = 'workout_logs.csv'
    engine = WorkoutEngine(csv_path)
    
    exercise_to_test = "Squats" 
    target_date = "2026-03-23"
    
    history = engine.get_exercise_history(exercise_to_test, up_to_date=target_date)
    if history is None:
        print(f"No data found for {exercise_to_test}.")
        return

    default_target, notes = engine.calculate_default_progression(history)
    
    print(f"--- {exercise_to_test.upper()} PIPELINE ---")
    print(f"Algorithm Target: {default_target['weight_kg']}kg for {default_target['reps']} ({default_target['action']})")
    print(f"User Notes: '{notes}'")
    
    needs_ai, trigger_reason = needs_ai_intervention(notes, history)
    
    if needs_ai:
        print(f"\n=> AI Triggered. Reason: {trigger_reason}")
        print("=> Consulting AI API...")
        try:
            ai_feedback = get_ai_coaching(exercise_to_test, history, default_target, notes, trigger_reason)
            
            if ai_feedback.is_override:
                print("\n⚠️ AI OVERRIDE ACTIVATED ⚠️")
                if ai_feedback.suggested_variation:
                    print(f"New Exercise: {ai_feedback.suggested_variation}")
                if ai_feedback.override_weight_kg:
                    print(f"New Target Weight: {ai_feedback.override_weight_kg}kg")
                if ai_feedback.override_reps:
                    print(f"New Target Reps: {ai_feedback.override_reps}")
            else:
                print("\n✅ AI Approved Algorithm Target.")
                
            print(f"\nCoach Feedback: {ai_feedback.coach_feedback}")
            
        except Exception as e:
            print(f"AI Call Failed: {e}")
    else:
        print(f"\n=> No AI needed. {trigger_reason} Cost: $0.00.")

if __name__ == "__main__":
    run_pipeline()