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
def fetch_plan_history(query_type: str, filepath='workout_plans.csv'):
    """Fetches the current active plan or the most recent previous plan."""
    try:
        plans = pd.read_csv(filepath)
        
        if query_type == "current":
            # Active plans have no end_date
            current_plan = plans[plans['end_date'].isna()]
            if current_plan.empty:
                return "No active plan found."
            
            # Group by day and sort by execution order
            structured_plan = current_plan.groupby('day_name').apply(
                lambda x: x.sort_values('order')[['order', 'exercise']].to_dict('records')
            ).to_dict()
            
            return {
                "status": "current", 
                "start_date": current_plan['start_date'].iloc[0], 
                "schedule": structured_plan
            }
            
        elif query_type == "previous":
            # Past plans have an end_date
            past_plans = plans[plans['end_date'].notna()]
            if past_plans.empty:
                return "No previous plans found."
            
            # Find the most recently ended plan
            last_plan_id = past_plans.sort_values('end_date', ascending=False).iloc[0]['plan_id']
            last_plan_data = past_plans[past_plans['plan_id'] == last_plan_id]
            
            structured_plan = last_plan_data.groupby('day_name').apply(
                lambda x: x.sort_values('order')[['order', 'exercise']].to_dict('records')
            ).to_dict()
            
            return {
                "status": "previous",
                "plan_id": last_plan_id,
                "start_date": last_plan_data['start_date'].iloc[0],
                "end_date": last_plan_data['end_date'].iloc[0],
                "schedule": structured_plan
            }
    except Exception as e:
        return f"Database Error: {e}"
import csv

def fetch_session_context(target_date: str) -> dict:
    """Retrieves the global pre-flight context for a specific workout date."""
    try:
        with open('session_history.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['date'] == target_date:
                    return row
    except FileNotFoundError:
        print("⚠️ session_history.csv not found.")
    
    # Return a default 'normal' state if the date isn't found
    return {
        "sleep": "unknown", "fatigue": "unknown", "physical_activity": "unknown",
        "nutrition": "unknown", "stress": "unknown", "soreness_pain": "unknown", 
        "other_notes": "No data recorded for this session."
    }

# ==========================================
# 3. THE AGENTIC LOOP
# ==========================================

import json
from pydantic import TypeAdapter

def call_ai_coach(payload, client):
    """The interactive ReAct state machine supporting multi-action turns."""
    print("\n[INITIALIZING AI AGENT...]")
    
    agent_adapter = TypeAdapter(AgentResponse)
    schema_instructions = json.dumps(agent_adapter.json_schema(), indent=2)
    valid_exercises = get_user_exercise_catalog()
    
    system_prompt = f"""
You are an expert strength coach investigating a user's stalled progress,
performance decline, or reported training problem.

Your job is NOT to immediately prescribe a solution. Your job is to
investigate the available evidence, form plausible hypotheses, gather
the most useful missing information, and only then decide whether an
intervention is justified.

You have four tools available:

1. "QUERY_DATABASE"
   Look up recent performance history for a specific exercise.

2. "ASK_USER"
   Ask the user for information that is not available in the database,
   such as sleep, fatigue, pain characteristics, recent activities,
   nutrition, stress, or upcoming training/recovery circumstances.

3. "QUERY_PLAN"
   Look up the user's current or previous workout plan.

4. "FINALIZE_DIAGNOSIS"
   Finish the investigation and provide the most appropriate recommendation.
   A recommendation does NOT necessarily mean changing the program.
   If the evidence suggests the problem is temporary or isolated, you may
   recommend continuing the existing program without modification.
   remembeer to provide an output for the issue log , irresepctive of whether the diagnosis is a change or not.

--------------------------------------------------
INVESTIGATION PRINCIPLES
--------------------------------------------------

1. RETRIEVE BEFORE ASKING
If the information may already exist in the database or workout-plan
history, retrieve it before asking the user.

The user's baseline lifestyle factors for this specific workout are ALREADY 
provided in the initial payload under `global_session_context`.
- You MUST read this context first. 
- Do NOT use the ASK_USER tool to ask about sleep, nutrition, general fatigue, 
  overall stress, or general soreness. That data is already in front of you.
- ONLY use the ASK_USER tool if you need highly specific mechanical details 
  (e.g., "Where exactly in the elbow does it hurt during the pushdown?"). or if you need to clarify a specific recent event that may have affected training. or if you need more info regarding the user's lifestyle context that is not already provided in the initial payload.

2. DO NOT FORCE A DIAGNOSIS
If the evidence is insufficient, continue investigating.
Explicitly distinguish between:
- strongly supported explanation
- plausible explanation
- unresolved possibility

3. INVESTIGATE PATTERNS, NOT JUST INDIVIDUAL EXERCISES
When an exercise stalls or declines, determine whether the problem is isolated
or affecting several related lifts. You may query multiple relevant exercises
in a single turn to evaluate this pattern efficiently.

4. ALWAYS CONSIDER RECENT PROGRAM CHANGES
Use QUERY_PLAN to inspect current or previous structures when relevant.

5. CONSIDER TEMPORARY VS PERSISTENT PROBLEMS
A single bad session does not automatically justify changing targets.
Distinguish acute flukes from chronic trends.

6. EVERY PRESCRIPTION MUST BE EVIDENCE-BASED
Make sure you have investigated an exercise's history before adjusting it.

7. PAIN REQUIRES CAUTION
Use language like "may be contributing" or "is consistent with".
Never claim a definitive medical diagnosis.

--------------------------------------------------
DATABASE DIRECTORY
--------------------------------------------------
When using QUERY_DATABASE, you MUST select the exact exercise name from:
{valid_exercises}

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------
You MUST respond in strict JSON format matching this exact schema:
{schema_instructions}

MULTI-QUERY BATCHING:
You may include MULTIPLE actions in the "actions" array if you need data on
multiple exercises or need both the database and plan details at once.
""" 
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Initial Payload: {json.dumps(payload)}"}
    ]

    while True:
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b", 
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content
            
            # 1. ALWAYS append the assistant's output to retain dialogue history
            messages.append({"role": "assistant", "content": response_content})
            
            # 2. Parse and validate through Pydantic
            parsed_envelope = agent_adapter.validate_json(response_content)
            actions_list = parsed_envelope.actions

            system_results = []
            ask_user_encountered = False

            for action in actions_list:
                if action.action_type == "QUERY_DATABASE":
                    print(f"🔍 AI is looking up history for: {action.exercise_to_query}")
                    data = fetch_exercise_history(action.exercise_to_query)
                    system_results.append(f"Database ({action.exercise_to_query}): {data}")
                    
                elif action.action_type == "QUERY_PLAN":
                    print(f"📋 AI is looking up plan details: {action.query_type.upper()}")
                    data = fetch_plan_history(action.query_type)
                    system_results.append(f"Plan ({action.query_type}): {data}")
                    
                elif action.action_type == "ASK_USER":
                    print(f"🧠 AI Reasoning: {action.reasoning}")
                    print(f"🗣️ COACH: {action.question}")
                    user_answer = input("👉 YOUR ANSWER: ")
                    system_results.append(f"USER ANSWER: {user_answer}")
                    ask_user_encountered = True
                    # Stop batch if user interaction is needed
                    break 
                    
                elif action.action_type == "FINALIZE_DIAGNOSIS":
                    # print("\n✅ FINAL AI PRESCRIPTION:")
                    # print(f"Diagnosis: {action.analysis}")
                    # for adj in action.adjustments:
                    #     print(f"  - {adj.exercise} Target: {adj.new_target_weight_kg}kg x {adj.new_target_reps} @ RIR {adj.new_target_rir}")
                    #     print(f"    Cue: {adj.ai_instructions}")
                    return action.model_dump()

            # 3. Feed gathered results back to the agent in a single turn
            if system_results:
                combined_results = "\n---\n".join(system_results)
                messages.append({
                    "role": "user",
                    "content": f"SYSTEM TOOL RESULTS:\n{combined_results}"
                })
                
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
        daily_context = fetch_session_context(target_date) 
        for mg, flags in grouped_issues.items():
            print(f"\nGathering context for muscle group: [{mg}]...")
            flagged_names = [f['exercise'] for f in flags]
            history = get_issue_history(mg)
            
            ai_payload = {
                "muscle_group": mg,
                "flagged_exercises": flags,
                "global_session_context": daily_context,
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
        run_pipeline('2026-01-27', api_key)