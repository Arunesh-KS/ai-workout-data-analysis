import pandas as pd
import json
import os
import time
from groq import Groq

# Import your cleanly separated logic
from engine import ProgressionEngine
from pydantic import TypeAdapter
from schema import AgentResponse  # Schema without InvestigationState


# ==========================================
# CUSTOM JSON ENCODER (Handles NumPy/Pandas types)
# ==========================================
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


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
    
    #mg_issues = issues[issues['muscle_group'] == muscle_group].fillna("")
    mg_issues=issues
    active = issues[mg_issues['status'] == 'Active']
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
        
    session['set_index'] = session.groupby('exercise').cumcount() + 1
    flagged_tuples = {(item['exercise'], item['set_index']) for item in flagged_queue_items}
    
    healthy_sets = []
    for _, row in session.iterrows():
        if (row['exercise'], row['set_index']) not in flagged_tuples:
            healthy_sets.append({
                'exercise': row['exercise'],
                'set_index': row['set_index'],
                'weight': float(row['weight']),
                'reps': int(row['reps']),
                'rir': int(row['rir'])
            })
            
    return healthy_sets

def fetch_exercise_history(exercise_name: str, filepath='workout_logs.csv', session_limit=3):
    """Fetches all set logs for an exercise across the last N distinct workout sessions."""
    try:
        logs = pd.read_csv(filepath)
        ex_logs = logs[logs['exercise'] == exercise_name]
        if ex_logs.empty:
            return f"No recent data found for {exercise_name}."
            
        # Get unique dates sorted chronologically, then take the last N sessions
        unique_dates = sorted(ex_logs['date'].dropna().unique())
        last_dates = unique_dates[-session_limit:]
        
        recent_logs = ex_logs[ex_logs['date'].isin(last_dates)]
        return recent_logs.to_dict(orient='records')
    except Exception as e:
        return f"Database Error: {e}"

def fetch_plan_history(query_type: str, filepath='workout_plans.csv'):
    """Fetches the current active plan or the most recent previous plan."""
    try:
        plans = pd.read_csv(filepath)
        plan_cols = [
            'order', 'exercise', 'lower_reps', 'upper_reps', 
            'rir', 'weight_increment(kg)', 'current_target_weight(kg)', 'current_target_reps'
        ]
        
        if query_type == "current":
            current_plan = plans[plans['end_date'].isna()]
            if current_plan.empty:
                return "No active plan found."
            
            structured_plan = current_plan.groupby('day_name').apply(
                lambda x: x.sort_values('order')[plan_cols].to_dict('records')
            ).to_dict()
            
            return {
                "status": "current", 
                "start_date": current_plan['start_date'].iloc[0], 
                "schedule": structured_plan
            }
            
        elif query_type == "previous":
            past_plans = plans[plans['end_date'].notna()]
            if past_plans.empty:
                return "No previous plans found."
            
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
    
    return {
        "sleep": "unknown", "fatigue": "unknown", "physical_activity": "unknown",
        "nutrition": "unknown", "stress": "unknown", "soreness_pain": "unknown", 
        "other_notes": "No data recorded for this session."
    }
import re

def prune_exercise_history(messages, exercise_name: str, replace_note: str):
    """
    Targets the specific exercise inside 'pre_fetched_exercise_history' within the JSON string 
    and replaces its raw data array/dict with a concise replace note.
    """
    found_any = False
    for msg in messages[1:]:
        if "content" in msg and isinstance(msg["content"], str):
            # Target the key-value pair for this specific exercise inside the JSON structure
            # e.g., "squats": [ ... ] or "squats": { ... }
            pattern = rf'(["\']?{exercise_name}["\']?\s*:\s*(?:\[.*?\]|\{{.*?\}}))'
            
            replacement_text = f'"{exercise_name}": "[PRUNED: {replace_note}]"'
            
            new_content, count = re.subn(pattern, replacement_text, msg["content"], flags= re.DOTALL | re.IGNORECASE)
            if count > 0:
                msg["content"] = new_content
                found_any = True
                
    if found_any:
        print(f"🧹 JSON-pruned history for exercise '{exercise_name}'.")
    else:
        messages[-1]["content"] += f"\n[NOTE ON {exercise_name.upper()}]: {replace_note}"
        print(f"⚠️ Exercise '{exercise_name}' key not found in JSON payload; appended note.")

def prune_previous_plan(messages, replace_note: str):
    """
    Targets 'pre_fetched_current_plan' (or general plan keys) in the JSON payload 
    and clears its bulky data.
    """
    found_any = False
    for msg in messages[1:]:
        if "content" in msg and isinstance(msg["content"], str):
            # Target the pre_fetched_current_plan key in the JSON
            pattern = r'(["\']?pre_fetched_current_plan["\']?\s*:\s*(?:\[.*?\]|\{{.*?\}}|null|["\'].*?["\']))'
            
            replacement_text = f'"pre_fetched_current_plan": "[PRUNED PREVIOUS PLAN: {replace_note}]"'
            
            new_content, count = re.subn(pattern, replacement_text, msg["content"], flags=re.DOTALL | re.IGNORECASE)
            if count > 0:
                msg["content"] = new_content
                found_any = True
                
    if found_any:
        print(f"🧹 JSON-pruned previous plan data.")
    else:
        messages[-1]["content"] += f"\n[PREVIOUS PLAN NOTE]: {replace_note}"
        print(f"⚠️ 'pre_fetched_current_plan' key not found in JSON payload; appended note.")

import json

def filter_issue_history(messages, relevant_ids: list[int], replace_note: str):
    """
    Filters issue history across the message stack, keeping only the IDs specified 
    in relevant_ids and replacing the rest with the replace_note.
    """
    found_any = False
    
    for msg in messages[1:]:
        if "content" in msg and isinstance(msg["content"], str):
            try:
                # Try to parse the message content as JSON if it represents the full payload dict
                payload = json.loads(msg["content"])
                
                # Check if issue keys exist in this payload
                if "active_issues" in payload or "past_rectified_issues" in payload:
                    # Filter active issues
                    if "active_issues" in payload:
                        payload["active_issues"] = [
                            issue for issue in payload["active_issues"] 
                            if int(issue.get("id", -1)) in relevant_ids
                        ]
                    
                    # Filter past rectified issues
                    if "past_rectified_issues" in payload:
                        payload["past_rectified_issues"] = [
                            issue for issue in payload["past_rectified_issues"] 
                            if int(issue.get("id", -1)) in relevant_ids
                        ]
                    
                    # Attach the explanatory note
                    payload["issue_history_filter_note"] = replace_note
                    
                    # Serialize back to the message content string
                    msg["content"] = json.dumps(payload, indent=2)
                    found_any = True
            except json.JSONDecodeError:
                # Fallback if the message isn't pure JSON
                continue
                
    if found_any:
        print(f"🧹 Successfully filtered issue history keeping IDs: {relevant_ids}")
    else:
        messages[-1]["content"] += f"\n[ISSUE HISTORY FILTER NOTE]: {replace_note}"
        print(f"⚠️ Could not parse JSON payload for issue filtering; appended note.")
# ==========================================
# 3. THE AGENTIC LOOP (MESSAGE APPEND MODE)
# ==========================================

def call_ai_coach(initial_user_message,ai_payload, client):
    """The interactive ReAct state machine using history-rich message appending."""
    print("\n[INITIALIZING AI AGENT (MESSAGE APPEND)...]")
    
    agent_adapter = TypeAdapter(AgentResponse)
    schema_instructions = json.dumps(agent_adapter.json_schema(), indent=2)
    valid_exercises = get_user_exercise_catalog()
    
    system_prompt = f"""
    You are an expert strength coach investigating stalled progress, performance decline, or reported training problems.
Do not prescribe immediately. First inspect the available evidence, form plausible hypotheses, gather the most useful missing information, then decide whether intervention is justified.

AVAILABLE COMMANDS

1. `QUERY_DATABASE`
   Look up recent performance history for an exercise.
   Flagged-exercise history is already provided in `pre_fetched_exercise_history`, so do not query those exercises again. Use this only for additional relevant exercises.

2. `ASK_USER`
   Ask for specific information not available in the provided data.

3. `QUERY_PLAN`
   Query the user's previous workout plan when relevant. The current plan is already provided, so do not query it.

4. `FINALIZE_DIAGNOSIS`
   Finish the investigation and prescribe target adjustments or maintain current targets.

5. `REMOVE_HISTORY`
   Remove exercise history from the active context only after determining it is no longer relevant. Always provide a concise `replace_note` explaining why it can be removed. Do not remove it prematurely.

6. `REMOVE_PREVIOUS_PLAN`
   Remove the previous plan only after the plan comparison is complete and it is no longer relevant. Always provide a concise `replace_note`.

INVESTIGATION RULES

1. Retrieve before asking.
   If information may already exist in the database or plan history, retrieve it before asking the user. Do not ask for information available through tools.

The initial payload already contains `global_session_context`.
Do not ask about sleep, nutrition, general fatigue, stress, or general soreness. Ask only for specific mechanical details, recent events, or lifestyle information not already provided.

Before using `QUERY_DATABASE`, check `pre_fetched_exercise_history`. If the exercise is already there, use that data instead of querying it.

2. Do not force a diagnosis.
   If evidence is insufficient, continue investigating. Distinguish between:

* strongly supported
* plausible
* unresolved

3. Investigate patterns, not only individual exercises.
   Determine whether the problem is isolated or affects related exercises. Independent database queries may be batched.

4. Distinguish temporary from persistent problems.
   One bad session does not automatically justify changing targets. do not change target if the issue is external factors like sleep , nutrition , change in gym env , grip issue / machine issues , etc . act rationally and change only when required .

5. Plan set/order handling.
   In the workout plan:

* `order` = global exercise/set sequence within the workout
* `set_index` = local set number for that exercise

Different sets of the same exercise may have different targets, reps, weights, or RIR. Always reference the correct `set_index` when making adjustments. A problem may affect one set or all sets.

6. Context removal.
   When removing exercise history, issue history, or the previous plan, always provide a concise `replace_note`, to explain why the data is being removed , and conclusions , etc . Remove data only after determining it is no longer relevant.
   if you are handling multiple exercises, you may remove the history of exercises for which you have already determined the cause , which you could add to the replace_note .
DATABASE DIRECTORY

When using `QUERY_DATABASE`, use the exact exercise name from:
`{valid_exercises}`

OUTPUT FORMAT

Respond in strict JSON matching:
`{schema_instructions}`

MULTI-QUERY BATCHING

You may include multiple independent actions in `actions`, such as:
QUERY_DATABASE for multiple exercises, QUERY_PLAN , ASK_USER. However, do not batch actions when one depends on the result of another.
Do not batch actions when one depends on the result of another.
"""
    

            # Initialize messages array with this XML string
    messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_user_message}
            ]
   
    
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    interaction_count = 0
    
    while True:
        try:
            time.sleep(5) # Throttle to respect rate limits
            
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b", 
                messages=messages,
                temperature=0.2,
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

            system_results = []

            # 2. Execute Actions
            for action in actions_list:
                if action.action_type == "QUERY_DATABASE":
                    print(f"🔍 AI is looking up history for: {action.exercise_to_query}")
                    data = fetch_exercise_history(action.exercise_to_query)
                    system_results.append(f"Database ({action.exercise_to_query}): {json.dumps(data, cls=CustomJSONEncoder)}")
                    
                elif action.action_type == "QUERY_PLAN":
                    print(f"📋 AI is looking up plan details: {action.query_type.upper()}")
                    data = fetch_plan_history(action.query_type)
                    system_results.append(f"Plan ({action.query_type}): {json.dumps(data, cls=CustomJSONEncoder)}")
                    
                elif action.action_type == "ASK_USER":
                    print(f"🧠 AI Reasoning: {action.rationale}") 
                    print(f"🗣️ COACH: {action.question}")
                    user_answer = input("👉 YOUR ANSWER: ")
                    system_results.append(f"USER ANSWER: {user_answer}")
                    break 
                    
                elif action.action_type == "FINALIZE_DIAGNOSIS":
                    print("total tokens used in this session: ", total_tokens)
                    print(f"Total Input Tokens: {total_input_tokens}")
                    print(f"Total Output Tokens: {total_output_tokens}")
                    print(f"Number of User Interactions: {interaction_count}")
                    return action.model_dump()
                elif action.action_type == "REMOVE_HISTORY":
                    prune_exercise_history(messages, action.exercise, action.replace_note)
                    system_results.append(f"System: Successfully removed raw history for {action.exercise}.")

                elif action.action_type == "REMOVE_PREVIOUS_PLAN":
                    prune_previous_plan(messages, action.replace_note)
                    system_results.append(f"System: Successfully removed previous plan details.")

            # 3. 📜 MESSAGE APPEND: Append assistant turn and tool results to history
            if system_results:
                combined_results = "\n---\n".join(system_results)
                
                # Append assistant response to preserve history
                messages.append({"role": "assistant", "content": response_content})
                # Append tool/user feedback as next user message
                messages.append({"role": "user", "content": f"SYSTEM RESULTS / USER UPDATE:\n{combined_results}"})
                
                print("📜 [History Appended to Message Stack]")
                
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
            
            # Pre-fetch history for every flagged exercise
            pre_fetched_histories = {}
            flagged_exercise_names = []
            for flag in flags:
                ex_name = flag['exercise']
                #print(f"Pre-fetching history for flagged exercise: {ex_name}")
                if ex_name not in flagged_exercise_names:
                    flagged_exercise_names.append(ex_name)
                    pre_fetched_histories[ex_name] = fetch_exercise_history(ex_name, session_limit=3)

            # Pre-fetch the current workout plan
            current_plan_data = fetch_plan_history("current")

            ai_payload = {
                "muscle_group": mg,
                "flagged_sets": flags, 
                "healthy_sets_today": get_healthy_sets(target_date, mg, 'workout_logs.csv', flags),
                "pending_fast_path_updates": fast_path_updates,
                "global_session_context": daily_context,
                
                # Pre-fetched data injected directly into Turn 1!
                "pre_fetched_current_plan": current_plan_data,
                "pre_fetched_exercise_history": pre_fetched_histories,
                
                "active_issues": history["active_issues"],
                "past_rectified_issues": history["past_rectified_issues"]
            }
            # Create the XML-formatted initial message
            initial_user_message = f"""
            <session_context date="{target_date}" muscle_group="{mg}">
            
              <data_inventory_manifest>
                CRITICAL NOTICE: The complete history and current plan data for the following flagged exercises are ALREADY loaded below. 
                You DO NOT need to call QUERY_DATABASE or QUERY_PLAN for them:
                {chr(10).join(f"- {name}" for name in flagged_exercise_names)}
                
                *Note: You may still use QUERY_DATABASE if the user introduces new symptoms that require checking OTHER unlisted auxiliary exercises.*
              </data_inventory_manifest>
            
              <global_lifestyle_context>
                {json.dumps(daily_context, indent=2)}
              </global_lifestyle_context>
            
              <flagged_sets>
                {json.dumps(flags, indent=2)}
              </flagged_sets>
            
              <pre_fetched_current_plan>
                {json.dumps(current_plan_data, indent=2)}
              </pre_fetched_current_plan>
            
              <pre_fetched_exercise_history>
                {json.dumps(pre_fetched_histories, indent=2)}
              </pre_fetched_exercise_history>
            
              <active_issues>
                {json.dumps(history["active_issues"], indent=2)}
              </active_issues>
            
            </session_context>
            """
            # print("\n" + "="*40 + " INITIAL PAYLOAD DEBUG " + "="*40)
            # print(initial_user_message)
            # print("="*103 + "\n")
            ai_prescription = call_ai_coach(initial_user_message, ai_payload, client)
        
            
            
            
            if ai_prescription:
                print("\n✅ FINAL AI PRESCRIPTION:")
                print(f"  Diagnosis: {ai_prescription['analysis']}")
                for adj in ai_prescription['adjustments']:
                    set_idx = adj.get('set_index', 'N/A')
                    print(f"  - {adj['exercise']} (Set {set_idx}) Target Updated: {adj['new_target_weight_kg']}kg x {adj['new_target_reps']} @ RIR {adj['new_target_rir']}")
                    print(f"  - Cue: {adj['ai_instructions']}")
                    
                print("\n📝 PROPOSED ISSUE LOG UPDATES:")
                for log_update in ai_prescription.get('issue_log_updates', []):
                    generated_advice = log_update.get('ai_advice', 'No advice recorded.')
                    print(f"  - APPEND TO issue_log.csv: {target_date} | {log_update.get('exercise')} | {mg} | {log_update.get('issue_type', 'Problem')} | {log_update.get('description', '')} | Active | \"{generated_advice}\"")

    else:
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
        run_pipeline('2026-01-27', api_key)

#very poor,very high,high,poor,high,high,Poor recovery and noticeable soreness before training.