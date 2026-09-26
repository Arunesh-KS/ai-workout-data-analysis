import pandas as pd
import json
import os
from groq import Groq

# Import your cleanly separated logic
from engine import ProgressionEngine
from pydantic import TypeAdapter
from schema import AgentResponse  # <-- This is all you need


# ==========================================
# 2. DATABASE HELPER FUNCTIONS
# ==========================================
def get_user_exercise_catalog(filepath='workout_logs.csv'):
    """Extracts a list of all exercises this specific user has ever performed."""
    try:
        logs = pd.read_csv(filepath)
        return logs['exercise'].dropna().unique().tolist()
    except FileNotFoundError:
        return []
def get_issue_history(muscle_group, filepath='issue_log.csv'):
    try:
        issues = pd.read_csv(filepath)
    except FileNotFoundError:
        return {"active_issues": [], "past_rectified_issues": []}
    
    mg_issues = issues[issues['muscle_group'] == muscle_group].fillna("")
    active = mg_issues[mg_issues['status'] == 'Active']
    rectified = mg_issues[mg_issues['status'] == 'Rectified']
    
    return {
        "active_issues": active.to_dict(orient='records'),
        "past_rectified_issues": rectified.to_dict(orient='records')
    }

def get_healthy_exercises(target_date, muscle_group, logs_path, flagged_exercises):
    logs = pd.read_csv(logs_path)
    session_logs = logs[(logs['date'] == target_date) & (logs['muscle_group'] == muscle_group)]
    all_exercises_today = session_logs['exercise'].unique()
    return [ex for ex in all_exercises_today if ex not in flagged_exercises]

def fetch_exercise_history(exercise_name: str, filepath='workout_logs.csv', limit=3):
    """New helper for the AI to query historical performance of any exercise."""
    try:
        logs = pd.read_csv(filepath)
        ex_logs = logs[logs['exercise'] == exercise_name].tail(limit)
        if ex_logs.empty:
            return f"No recent data found for {exercise_name}."
        return ex_logs.to_dict(orient='records')
    except Exception as e:
        return f"Database Error: {e}"

# ==========================================
# 3. THE AGENTIC LOOP
# ==========================================

def call_ai_coach(payload, client):
    """The interactive ReAct state machine."""
    print("\n[INITIALIZING AI AGENT...]")
    
    # We dynamically inject the JSON schema requirements so the LLM knows its boundaries
    agent_adapter = TypeAdapter(AgentResponse)
    
    # 2. Extract the schema using the adapter
    schema_instructions = json.dumps(agent_adapter.json_schema(), indent=2)
    valid_exercises = get_user_exercise_catalog()
    system_prompt = f"""
    You are an expert strength coach. You are diagnosing a user's stalled progress , or problems mentioned by them (if any).
    You have three tools available:
    1. "QUERY_DATABASE": Look up recent performance for other exercises.
    2. "ASK_USER": Ask clarifying questions about sleep, fatigue, or pain. make sure to diagnose the actual issue with confidence before finalizing the prescription.
    3. "FINALIZE_DIAGNOSIS": Use when 100% confident to adjust the program. you needn't always adjust the program , sometimes it might be just 1 bad day and the user can continue with the same program. If you are not confident, use QUERY_DATABASE or ASK_USER first. do this how a human coach would do it. 
    
    DATABASE DIRECTORY:
    When using QUERY_DATABASE, you MUST select the exact exercise name from this list:
    {valid_exercises}
    
    COACHING DIRECTIVE: If a user reports joint pain or secondary muscle fatigue (e.g., lower back during squats), you MUST use QUERY_DATABASE to check the history of related heavy compound lifts before asking the user.
    
    You MUST respond in strict JSON format matching this exact schema:
    {schema_instructions}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Initial Payload: {json.dumps(payload)}"}
    ]

    while True:
        try:
            # Note: Groq runs llama3-70b-8192 exceptionally well for JSON mode
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b", 
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw_json = response.choices[0].message.content
            
            # Pydantic validation handles the routing automatically
            ai_action = agent_adapter.validate_json(raw_json)
            
            # Save the AI's response to the chat history
            messages.append({"role": "assistant", "content": raw_json})

            # --- ROUTER ---
            if ai_action.action_type == "QUERY_DATABASE":
                print(f"\n🔍 AI is looking up history for: {ai_action.exercise_to_query}")
                print(f"   Reasoning: {ai_action.reasoning}")
                db_result = fetch_exercise_history(ai_action.exercise_to_query)
                messages.append({
                    "role": "user", 
                    "content": f"SYSTEM DATABASE RESULT ({ai_action.exercise_to_query}): {json.dumps(db_result)}"
                })
                
            elif ai_action.action_type == "ASK_USER":
                print(f"\n🧠 AI Reasoning: {ai_action.reasoning}")
                print(f"🗣️  COACH: {ai_action.question}")
                user_answer = input("👉 YOUR ANSWER: ")
                messages.append({
                    "role": "user", 
                    "content": f"USER ANSWER: {user_answer}"
                })
                
            elif ai_action.action_type == "FINALIZE_DIAGNOSIS":
                # Convert the Pydantic object back into a standard dictionary 
                # so the rest of your original run_pipeline code doesn't break.
                return ai_action.model_dump()
                
        except Exception as e:
            print(f"Agent Loop Error: {e}")
            return None

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def run_pipeline(target_date, api_key):
    client = Groq(api_key=api_key)
    engine = ProgressionEngine()
    
    print(f"\n--- RUNNING SESSION ANALYSIS FOR {target_date} ---")
    fast_path_updates, ai_routing_queue = engine.sweep_session(target_date)
    
    if fast_path_updates:
        print("\n✅ FAST PATH (DETERMINISTIC INCREMENTS):")
        for update in fast_path_updates:
            print(f"  - {update['exercise']}: target updated to {update['target_weight_kg']}kg x {update['target_reps']} @ RIR {update['target_rir']}")
            
    if ai_routing_queue:
        print("\n⚠️ AI INTERVENTION REQUIRED:")
        
        grouped_issues = {}
        for item in ai_routing_queue:
            mg = item['muscle_group']
            if mg not in grouped_issues:
                grouped_issues[mg] = []
            grouped_issues[mg].append(item)
            
        for mg, flags in grouped_issues.items():
            print(f"\nGathering context for muscle group: [{mg}]...")
            flagged_names = [f['exercise'] for f in flags]
            history = get_issue_history(mg)
            
            ai_payload = {
                "muscle_group": mg,
                "flagged_exercises": flags,
                "healthy_exercises_today": get_healthy_exercises(target_date, mg, 'workout_logs.csv', flagged_names),
                "active_issues": history["active_issues"],
                "past_rectified_issues": history["past_rectified_issues"]
            }
            
            ai_prescription = call_ai_coach(ai_payload, client)
            
            if ai_prescription:
                print("\n✅ FINAL AI PRESCRIPTION:")
                print(f"  Diagnosis: {ai_prescription['analysis']}")
                for adj in ai_prescription['adjustments']:
                    print(f"  - {adj['exercise']} Target Updated: {adj['new_target_weight_kg']}kg x {adj['new_target_reps']} @ RIR {adj['new_target_rir']}")
                    print(f"  - Cue: {adj['ai_instructions']}")
                    
                print("\n📝 PROPOSED ISSUE LOG UPDATES:")
                for adj in ai_prescription['adjustments']:
                    log_append_data = adj.get('issue_log_append')
                    if log_append_data:
                        generated_advice = log_append_data.get('ai_advice', 'No advice recorded.')
                        print(f"  - APPEND TO issue_log.csv: {target_date} | {adj['exercise']} | {mg} | {log_append_data.get('issue_type', 'Problem')} | {log_append_data.get('description', '')} | Active | \"{generated_advice}\"")

if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not found.")
        print("Run this in your terminal first: $env:GROQ_API_KEY=\"your_key_here\"")
    else:
        run_pipeline('2026-03-10', api_key)