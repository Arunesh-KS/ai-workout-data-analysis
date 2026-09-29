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

def get_healthy_sets(target_date, muscle_group, logs_path, flagged_queue_items):
    """Returns all sets for a muscle group that successfully hit their targets."""
    try:
        logs = pd.read_csv(logs_path)
    except FileNotFoundError:
        return []
        
    session = logs[(logs['date'] == target_date) & (logs['muscle_group'] == muscle_group)].copy()
    
    if session.empty:
        return []
        
    # Recreate the set_index exactly as we did in the engine
    session['set_index'] = session.groupby('exercise').cumcount() + 1
    
    # Create a fast lookup of explicitly flagged sets: e.g., {('squats', 2), ('squats', 3)}
    flagged_tuples = {(item['exercise'], item['set_index']) for item in flagged_queue_items}
    
    healthy_sets = []
    for _, row in session.iterrows():
        # If this specific set wasn't flagged, it's healthy!
        if (row['exercise'], row['set_index']) not in flagged_tuples:
            healthy_sets.append({
                'exercise': row['exercise'],
                'set_index': row['set_index'],
                'weight': float(row['weight']),
                'reps': int(row['reps']),
                'rir': int(row['rir'])
            })
            
    return healthy_sets

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
        
        # Columns to extract based on the new row-per-set schema
        plan_cols = [
            'order', 'exercise', 'lower_reps', 'upper_reps', 
            'rir', 'weight_increment(kg)', 'current_target_weight(kg)', 'current_target_reps'
        ]
        
        if query_type == "current":
            # Active plans have no end_date
            current_plan = plans[plans['end_date'].isna()]
            if current_plan.empty:
                return "No active plan found."
            
            # Group by day and sort by execution order
            structured_plan = current_plan.groupby('day_name').apply(
                lambda x: x.sort_values('order')[plan_cols].to_dict('records')
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
                lambda x: x.sort_values('order')[plan_cols].to_dict('records')
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

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Automatically convert Pandas/NumPy int64 and float64 to native Python types
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

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
You must actively track the status of these specific cause IDs. 
Do not invent new IDs. Use exactly these strings:

1. "inadequate_recovery": Systemic lifestyle fatigue (poor sleep, high daily steps, caloric deficit, life stress).
2. "inadequate_stimulus": Volume is too low; the muscle isn't getting enough work to grow.
3. "training_load_mismatch": Volume/intensity is too high; CNS burnout or systemic overtraining.
4. "pain_soreness": Acute joint pain, injury, or severe lingering DOMS preventing force generation.
5. "technique_issue": Form breakdown, range-of-motion changes, grip slipping, or tempo alterations.
6. "program_change": Recent overarching changes to the plan structure (days split, frequency).
7. "exercise_order": The lift was moved later in the session and is suffering from cumulative session fatigue.
8. "local_interference": A preceding exercise severely fatigued the specific prime movers for this lift.
9. "target_mismatch": The predicted target was simply unrealistic or mathematically miscalculated.
10. "measurement_issue": Logging error, skipped exercise, or equipment variation (e.g., different machine).

--------------------------------------------------
INVESTIGATION STATE MANAGEMENT (YOUR WHITEBOARD)
--------------------------------------------------
On EVERY turn, you must output an updated, ultra-compact `investigation_state` using these exact rules:
- `active_hypotheses`: List the exact cause IDs from above that are still plausible.
- `ruled_out_hypotheses`: List the cause IDs you have confidently eliminated based on evidence.
- `established_facts`: CRITICAL MEMORY. Write a dense bulleted list of the exact weights, reps, dates, plan changes, and user quotes you have retrieved. You MUST write the actual numbers here so you do not forget them and do not need to re-query the commands later.
- `unresolved_questions`: State exactly what specific information you still need to find out.

Remember to update the established_facts such that you do not need to re-query the commands or ask user later. Do not leave any important numbers out. You could give it a bit of context but do not write a long paragraph. Use bullet points and keep it concise. add all the tiny details you have learned so far. This is your CRITICAL MEMORY for the investigation.

--------------------------------------------------
AVAILABLE COMMANDS (ACTIONS)
--------------------------------------------------
Use these COMMANDS only when the information is not already available in the `global_session_context` or your established facts.
1. "QUERY_DATABASE": Look up recent performance history for a specific exercise.
2. "ASK_USER": Ask the user for highly specific information.
3. "QUERY_PLAN": Look up the user's "current" or "previous" workout plan.
4. "FINALIZE_DIAGNOSIS": Finish the investigation and prescribe target adjustments or maintain current targets. 
   *Note: Always provide an issue_log_update, even if the diagnosis requires no target changes.*

--------------------------------------------------
INVESTIGATION PRINCIPLES & OVERRIDES
--------------------------------------------------
1. RETRIEVE BEFORE ASKING
The user's baseline lifestyle factors are ALREADY provided in the initial payload under `global_session_context`.
- You MUST read this context first. 
- Do NOT use ASK_USER for general sleep, nutrition, or overall stress if it is already provided.
- ONLY use ASK_USER for highly specific mechanical details or to clarify an unresolved hypothesis.

2. ELIMINATION OVER GUESSING
If evidence strongly contradicts a cause, mark it as 'ruled_out' and add the contradiction to `evidence_against`. Narrow down the list until only the true root cause remains.

3. OVERRIDING THE FAST PATH (CRITICAL)
Your payload contains `pending_fast_path_updates`. These are the automatic weight progressions the system is planning to apply to healthy sets.
If you discover an injury, severe fatigue, or technique cheat that affects the entire session, you have the authority to issue target adjustments for sets that are currently in the `pending_fast_path_updates`. Any adjustment you make in FINALIZE_DIAGNOSIS will overwrite the Fast Path.
4. HOW TO READ THE PLAN DATA
When you use QUERY_PLAN, the schedule returns a list of items based on 'order'. 
CRITICAL RULE: Each individual item (row) in that schedule represents EXACTLY ONE PRESCRIBED SET. If you see two entries for "squats", that means the plan prescribes exactly 2 sets. Do NOT ask the user how many sets are prescribed; count the entries yourself. and remember , different sets of the same exercise may have different targets, so you must treat each set as a separate entity.
lower_reps and upper_reps are the target rep range for that specific set. weight_increment is the weight increment for that specific set. current_target_weight(kg) and current_target_reps are the exact targets for that set when higher reps are met . weight_increment(kg) is the amount to increase the target weight if the set is successful.
5. TRUST YOUR WHITEBOARD
Before adding a question to `unresolved_questions` or issuing a retrieval command, you MUST check if the answer is already in your `established_facts`. NEVER re-query the database or plan for information you already possess.


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
However:
- NEVER include ASK_USER in the same response as QUERY_DATABASE or QUERY_PLAN.
- NEVER include FINALIZE_DIAGNOSIS in the same response as QUERY_DATABASE, QUERY_PLAN, or ASK_USER.
- If additional retrieved information could affect what question should be asked, retrieve it first.
- After retrieval results are returned, reassess the investigation state before choosing the next command.

"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Initial Payload: {json.dumps(payload, cls=CustomJSONEncoder)}"}
    ]
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    interaction_count = 0
    
    while True:
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b", 
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

            # ==========================================
            # 👁️ PRINT THE AI'S MENTAL WHITEBOARD 
            # ==========================================
            print("\n" + "="*50)
            print("📝 CURRENT INVESTIGATION STATE:")
            print(investigation_state.model_dump_json(indent=2))
            print("="*50 + "\n")

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
                
                # Format the assistant memory so the model sees a complete envelope
                compressed_assistant_payload = json.dumps({
                    "investigation_state": investigation_state.model_dump(),
                    "actions": []  # Anchor the schema so it remembers actions are inside JSON
                }, indent=2)

                messages = [
                    messages[0],  # System Prompt
                    messages[1],  # Initial User Payload
                    {
                        "role": "assistant",
                        "content": compressed_assistant_payload
                    },
                    {
                        "role": "user",
                        "content": f"SYSTEM COMMAND RESULTS / USER UPDATE:\n{combined_results}"
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
    
    # We delay printing/committing Fast Path here so AI can potentially overwrite them.
            
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
            
            history = get_issue_history(mg)
            
            # Pass the flagged sets, healthy sets, and pending fast_path updates to the AI payload
            ai_payload = {
                "muscle_group": mg,
                "flagged_sets": flags, 
                "healthy_sets_today": get_healthy_sets(target_date, mg, 'workout_logs.csv', flags),
                "pending_fast_path_updates": fast_path_updates,
                "global_session_context": daily_context,
                "active_issues": history["active_issues"],
                "past_rectified_issues": history["past_rectified_issues"]
            }
            
            ai_prescription = call_ai_coach(ai_payload, client)
            
            if ai_prescription:
                print("\n✅ FINAL AI PRESCRIPTION:")
                print(f"  Diagnosis: {ai_prescription['analysis']}")
                for adj in ai_prescription['adjustments']:
                    # Note: We expect the AI to now include the set_index or order alongside the exercise
                    set_idx = adj.get('set_index', 'N/A')
                    print(f"  - {adj['exercise']} (Set {set_idx}) Target Updated: {adj['new_target_weight_kg']}kg x {adj['new_target_reps']} @ RIR {adj['new_target_rir']}")
                    print(f"  - Cue: {adj['ai_instructions']}")
                    
                print("\n📝 PROPOSED ISSUE LOG UPDATES:")
                for log_update in ai_prescription.get('issue_log_updates', []):
                    # We look at issue_log_updates explicitly to match your new schema
                    generated_advice = log_update.get('ai_advice', 'No advice recorded.')
                    print(f"  - APPEND TO issue_log.csv: {target_date} | {log_update.get('exercise')} | {mg} | {log_update.get('issue_type', 'Problem')} | {log_update.get('description', '')} | Active | \"{generated_advice}\"")

    else:
        # If there are no AI interventions, we can go ahead and print/commit the Fast Path
        if fast_path_updates:
            print("\n✅ FAST PATH (DETERMINISTIC INCREMENTS):")
            for update in fast_path_updates:
                print(f"  - {update['exercise']} (Set {update['order']}): target updated to {update['new_target_weight_kg']}kg x {update['new_target_reps']}")

if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not found.")
        print("Run this in your terminal first: $env:GROQ_API_KEY=\"your_key_here\"")
    else:
        run_pipeline('2026-01-25', api_key)

#very poor,very high,high,poor,high,high,Poor recovery and noticeable soreness before training.