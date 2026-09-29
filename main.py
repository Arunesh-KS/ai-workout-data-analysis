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
    You are an expert strength coach investigating a user's stalled progress,
performance decline, or reported training problem.

Your job is NOT to immediately prescribe a solution. Your job is to
investigate the available evidence, form plausible hypotheses, gather
the most useful missing information, and only then decide whether an
intervention is justified.


--------------------------------------------------
AVAILABLE COMMANDS
--------------------------------------------------
1. "QUERY_DATABASE": Look up recent performance history for a specific exercise. The history for flagged exercises and the current workout plan are ALREADY provided below under pre_fetched_exercise_history and pre_fetched_current_plan. Do NOT use QUERY_DATABASE or QUERY_PLAN for these exercises , so you can use this command only if you need to look up additional exercises that are relevant to the investigation.
2. "ASK_USER": Ask the user for highly specific information.
3. "QUERY_PLAN": Look up the user's "previous" workout plan , when relevant to the investigation . current plan is already provided in the initial payload, so you can use this command only if you need to look up the previous plan for context.
4. "FINALIZE_DIAGNOSIS": Finish the investigation and prescribe target adjustments or maintain current targets.
--------------------------------------------------
INVESTIGATION PRINCIPLES
--------------------------------------------------
    1. RETRIEVE BEFORE ASKING
If the information may already exist in the database or workout-plan
history, retrieve it before asking the user.
Do not ask the user for information that can be obtained through a tool.

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

8. in user's workout plan , the column called order corresponds to the global order of the exercise in the plan. The column called set_index corresponds to the local order of the set within that exercise on that day. Use these two columns carefully when referencing specific sets. note that different sets of the same exercise on the same day may have different target weights, reps, and RIR. Always reference the correct set by its local set_index when making adjustments . the stall may occur on one set of an exercise but not on another set of the same exercise. always reference the correct set by its local set_index when making adjustments.it may also occur on all the sets .

--------------------------------------------------
DATABASE DIRECTORY
--------------------------------------------------
When using QUERY_DATABASE, you MUST select the exact exercise name from:
{valid_exercises}

and more importantly , you must check the pre_fetched_exercise_history in the initial payload first before using QUERY_DATABASE for any of these exercises. if the exercise is already in pre_fetched_exercise_history, you must use that data instead of querying the database again.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------
You MUST respond in strict JSON format matching this exact schema:
{schema_instructions}

MULTI-QUERY BATCHING:
You may include MULTIPLE actions in the "actions" array if you need data on
multiple exercises or need both the database and plan details at once.
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
            time.sleep(2) # Throttle to respect rate limits
            
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b", 
                messages=messages,
                temperature=0.5,
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