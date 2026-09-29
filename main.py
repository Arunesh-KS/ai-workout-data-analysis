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

--------------------------------------------------
THE 10 POTENTIAL CAUSES (DIFFERENTIAL DIAGNOSIS)
--------------------------------------------------
You must actively track the status of these specific causes in your `investigation_state`. 
Do not invent new cause IDs. Use exactly these:

1. `inadequate_recovery`: Systemic lifestyle fatigue (poor sleep, high daily steps, caloric deficit, life stress).
2. `inadequate_stimulus`: Volume is too low; the muscle isn't getting enough work to grow.
3. `training_load_mismatch`: Volume/intensity is too high; CNS burnout or systemic overtraining.
4. `pain_soreness`: Acute joint pain, injury, or severe lingering DOMS preventing force generation.
5. `technique_issue`: Form breakdown, range-of-motion changes, grip slipping, or tempo alterations.
6. `program_change`: Recent overarching changes to the plan structure (days split, frequency).
7. `exercise_order`: The lift was moved later in the session and is suffering from cumulative session fatigue.
8. `local_interference`: A preceding exercise severely fatigued the specific prime movers for this lift.
9. `target_mismatch`: The predicted target was simply unrealistic or mathematically miscalculated.
10. `measurement_issue`: Logging error, skipped exercise, or equipment variation (e.g., different machine).

--------------------------------------------------
INVESTIGATION STATE MANAGEMENT (YOUR WHITEBOARD)
--------------------------------------------------
On EVERY turn, you must output an updated `investigation_state`:
- Update the `status` of each `potential_cause` (e.g., from 'possible' to 'ruled_out' or 'supported').
- Add concise bullet points to `supporting_evidence` or `evidence_against`.
- Append confirmed, undeniable truths to `established_facts`.
- Maintain a list of `unresolved_questions` that dictate your next tool calls.

--------------------------------------------------
AVAILABLE TOOLS (ACTIONS)
--------------------------------------------------
1. "QUERY_DATABASE": Look up recent performance history for a specific exercise.
2. "ASK_USER": Ask the user for highly specific information.
3. "QUERY_PLAN": Look up the user's "current" or "previous" workout plan.
4. "FINALIZE_DIAGNOSIS": Finish the investigation and prescribe target adjustments or maintain current targets. 
   *Note: Always provide an issue_log_update, even if the diagnosis requires no target changes.*

--------------------------------------------------
INVESTIGATION PRINCIPLES
--------------------------------------------------
1. RETRIEVE BEFORE ASKING
The user's baseline lifestyle factors are ALREADY provided in the initial payload under `global_session_context`.
- You MUST read this context first. 
- Do NOT use ASK_USER for general sleep, nutrition, or overall stress if it is already provided.
- ONLY use ASK_USER for highly specific mechanical details or to clarify an unresolved hypothesis.

2. ELIMINATION OVER GUESSING
If evidence contradicts a cause, mark it as 'ruled_out' and add the contradiction to `evidence_against`. Narrow down the list until only the true root cause remains.

3. PATTERNS OVER ISOLATION
When an exercise stalls, query multiple relevant exercises (e.g., all Push movements, or all leg movements) to see if the problem is systemic or isolated.

4. TEMPORARY VS PERSISTENT
Distinguish acute flukes (one bad day) from chronic trends (3 weeks of stalling).

--------------------------------------------------
DATABASE DIRECTORY
--------------------------------------------------
When using QUERY_DATABASE, you MUST select the exact exercise name from:
{valid_exercises}

--------------------------------------------------
CRITICAL FORMATTING & ANTI-HALLUCINATION RULES
--------------------------------------------------
You MUST respond in strict JSON format matching this exact schema:
{schema_instructions}

MULTI-QUERY BATCHING:
You may include MULTIPLE actions in the "actions" array if you need data on multiple exercises or plans simultaneously.

"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Initial Payload: {json.dumps(payload)}"}
    ]
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    interaction_count = 0
    
    while True:
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b", 
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            usage = response.usage
            total_tokens += usage.total_tokens
            total_input_tokens += usage.prompt_tokens
            total_output_tokens += usage.completion_tokens
            interaction_count += 1
            
            print(f"📊 Tokens: {usage.prompt_tokens} In | {usage.completion_tokens} Out | {usage.total_tokens} Total")
            
            response_content = response.choices[0].message.content
            
            # 1. Parse and validate through Pydantic
            parsed_envelope = agent_adapter.validate_json(response_content)
            actions_list = parsed_envelope.actions
            investigation_state = parsed_envelope.investigation_state

            system_results = []

            # 2. Execute Actions
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
                    # NOTE: Updated from .reasoning to .rationale to match your new schema!
                    print(f"🧠 AI Reasoning: {action.rationale}") 
                    print(f"🗣️ COACH: {action.question}")
                    user_answer = input("👉 YOUR ANSWER: ")
                    system_results.append(f"USER ANSWER: {user_answer}")
                    # Stop batch if user interaction is needed
                    break 
                    
                elif action.action_type == "FINALIZE_DIAGNOSIS":
                    print("total tokens used in this session: ", total_tokens)
                    print(f"Total Input Tokens: {total_input_tokens}")
                    print(f"Total Output Tokens: {total_output_tokens}")
                    print(f"Number of User Interactions: {interaction_count}")
                    return action.model_dump()

            # 3. 🗜️ STATE COMPRESSION: Rebuild messages array instead of appending
            if system_results:
                combined_results = "\n---\n".join(system_results)
                
                # Convert the Pydantic state model directly to a formatted JSON string
                state_json = investigation_state.model_dump_json(indent=2)
                
                messages = [
                    messages[0], # [0] Keep System Prompt
                    messages[1], # [1] Keep Initial User Payload
                    {
                        # [2] Inject the AI's internal whiteboard memory
                        "role": "assistant",
                        "content": f"CURRENT INVESTIGATION STATE:\n{state_json}"
                    },
                    {
                        # [3] Pass the new data it requested
                        "role": "user",
                        "content": f"SYSTEM TOOL RESULTS:\n{combined_results}"
                    }
                ]
                print("🗜️ [Context Compressed using Investigation State]")
                
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

#very poor,very high,high,poor,high,high,Poor recovery and noticeable soreness before training.