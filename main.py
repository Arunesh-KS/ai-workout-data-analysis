import pandas as pd
import json
from groq import Groq
from engine import ProgressionEngine

def get_active_issues(muscle_group, issues_path='issue_log.csv'):
    try:
        issues = pd.read_csv(issues_path)
        active = issues[(issues['muscle_group'] == muscle_group) & (issues['status'] == 'Active')]
        return active[['exercise', 'issue_type', 'description', 'date_flagged']].to_dict(orient='records')
    except FileNotFoundError:
        return []

def get_healthy_exercises(target_date, muscle_group, logs_path, flagged_exercises):
    logs = pd.read_csv(logs_path)
    session_logs = logs[(logs['date'] == target_date) & (logs['muscle_group'] == muscle_group)]
    all_exercises_today = session_logs['exercise'].unique()
    return [ex for ex in all_exercises_today if ex not in flagged_exercises]

def call_ai_coach(payload, client):
    """Sends the cross-contextual payload to the real LLM."""
    print("\n[AI COACH ANALYZING PAYLOAD...]")
    
    # We will build out a robust prompt later. 
    # For now, this just passes the raw JSON to test the connection.
    prompt = f"""
    You are an expert strength coach. Analyze the following workout data and provide a JSON response.
    The user's actual strength (e1rm) dropped below their target, or they reported a problem.
    1. your goal is to help the user resolve their problems if any , or if their strength declined , help them get back on track with a new progression/variation/suggestion to plan .
    2. Provide fitness advice considering the user's current performance/strength level, any reported issues, and historical context.
    3. analyze the flagged exercises with data on other exercises of the same muscle group to diagonize the issue properly and solve it .
    4. before you suggest a new progression , check if the user is stalling or if they are just having a bad day and provide a solution accordingly .
    5. If you suggest a new progression, provide the new target weight, reps, and RIR for the next session , like a professional coach would do. make sure the numbers are reasonable and not exxagerated , or too low . also provide cues for form and recovery if needed .
    Data: {json.dumps(payload, indent=2)}
    
    Respond STRICTLY with JSON matching this format:
    {{
        "analysis": "Brief biomechanical explanation of the issue.",
        "adjustments": [
            {{
                "exercise": "Exercise Name",
                "new_target_weight_kg": 0.0,
                "new_target_reps": 0,
                "new_target_rir": 0,
                "ai_instructions": "Brief cue or instruction."
            }}
        ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error: {e}")
        return None

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
            ai_payload = {
                "muscle_group": mg,
                "flagged_exercises": flags,
                "healthy_exercises_today": get_healthy_exercises(target_date, mg, 'workout_logs.csv', flagged_names),
                "historical_active_issues": get_active_issues(mg)
            }
            
            ai_prescription = call_ai_coach(ai_payload, client)
            
            if ai_prescription:
                print("\n🧠 AI PRESCRIPTION:")
                print(f"  Diagnosis: {ai_prescription['analysis']}")
                for adj in ai_prescription['adjustments']:
                    print(f"  - {adj['exercise']} Target Updated: {adj['new_target_weight_kg']}kg x {adj['new_target_reps']} @ RIR {adj['new_target_rir']}")
                    print(f"  - Cue: {adj['ai_instructions']}")
                    
                print("\n📝 PROPOSED ISSUE LOG UPDATES:")
                for f in flags:
                    print(f"  - APPEND TO issue_log.csv: {target_date} | {f['exercise']} | {mg} | {f['status']} | {f['problem_note']} | Active")

import os

if __name__ == "__main__":
    # Pull the key securely from the system's environment variables
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not found.")
        print("Run this in your terminal first: $env:GROQ_API_KEY=\"your_key_here\"")
    else:
        run_pipeline('2026-03-10', api_key)