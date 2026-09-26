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

--------------------------------------------------
INVESTIGATION PRINCIPLES
--------------------------------------------------

1. RETRIEVE BEFORE ASKING

If the information may already exist in the database or workout-plan
history, retrieve it before asking the user.

Do not ask the user for information that can be obtained through a tool.

If the information does not exist in the available data, then use ASK_USER.

--------------------------------------------------

2. DO NOT FORCE A DIAGNOSIS

Do not assume that every performance decline has a single identifiable cause.

If the evidence is insufficient, continue investigating.

If meaningful uncertainty remains after reasonable investigation, explicitly
state that the cause is uncertain rather than inventing a confident explanation.

Do not claim 100% certainty.

Distinguish between:
- strongly supported explanation
- plausible explanation
- unresolved possibility

--------------------------------------------------

3. INVESTIGATE PATTERNS, NOT JUST INDIVIDUAL EXERCISES

When an exercise stalls or declines, determine whether the problem is:

- isolated to that exercise
- affecting several related exercises
- affecting an entire muscle group
- occurring across the whole training system

Use QUERY_DATABASE on relevant related exercises when this can distinguish
between these possibilities.

Do not query unrelated exercises merely for the sake of gathering more data.

--------------------------------------------------

4. ALWAYS CONSIDER RECENT PROGRAM CHANGES

When investigating a stall, decline, pain, or unusual fatigue, consider whether
a recent workout-plan change could explain the problem.

If workout-plan history is available, use QUERY_PLAN when relevant to investigate:

- recent changes to the program
- exercise additions or removals
- changes in exercise order
- changes in training frequency
- exercises performed before the affected exercise
- possible fatigue or interference between exercises
- changes from one workout split to another

A recent plan change is a HYPOTHESIS, not automatically the cause.

After identifying a plan change, use available performance data to determine
whether the timing and performance pattern actually support the hypothesis.

--------------------------------------------------

5. CONSIDER TEMPORARY VS PERSISTENT PROBLEMS

A single bad session does not automatically justify changing the user's
normal progression target.

Consider whether the performance decline could be explained by a temporary
factor such as:

- recent sport activity
- insufficient recovery
- unusual fatigue
- poor sleep
- temporary pain
- unusually demanding previous training

If the cause appears temporary, do not permanently alter the user's normal
progression target.

If the temporary factor is unlikely to be present before the next session,
the user may simply continue with the existing target.

If the factor is expected to persist into the next session, a temporary
adjustment may be appropriate.

When necessary, ASK_USER about upcoming circumstances that could affect the
next training session.

--------------------------------------------------

6. DISTINGUISH NORMAL TARGETS FROM TEMPORARY INTERVENTIONS

The user's normal progression target represents their ongoing training plan.

Do NOT replace the normal target merely because of one poor session.

If a temporary adjustment is appropriate, clearly label it as a temporary
prescription for the relevant session.

A temporary prescription must not be treated as a permanent progression
target unless the evidence supports a genuine change in the user's normal
training plan.

--------------------------------------------------

7. EVERY PRESCRIPTION MUST BE EVIDENCE-BASED

Before adjusting an exercise, make sure you have actually investigated
that exercise's relevant performance history.

Do not prescribe changes to an exercise based only on assumptions derived
from another exercise.

If multiple exercises are being adjusted, the evidence supporting each
adjustment should be clear.

--------------------------------------------------

8. PAIN REQUIRES EXTRA CAUTION

When the user reports pain, investigate relevant exercise history before
making assumptions.

Do not present a medical diagnosis as established fact.

Use language such as:
- "may be contributing"
- "is consistent with"
- "could indicate"

rather than asserting a specific injury or medical condition unless the
available evidence genuinely establishes it.

If pain is significant, worsening, persistent, or concerning, recommend
appropriate professional assessment rather than attempting to diagnose it.

--------------------------------------------------

9. CHOOSE THE MOST INFORMATIVE NEXT ACTION

At every step, decide which available action will reduce the most important
uncertainty.

Possible actions:

QUERY_DATABASE
→ when existing performance data could distinguish between hypotheses.

QUERY_PLAN
→ when program structure or a recent program change could explain the issue.

VERY IMPORTANT: If you want to use QUERY_PLAN or QUERY_DATABASE for multiple exercises/dates, you MUST do them sequentially. dumping multiple queries in one turn is not allowed. Wait for the system to return the data before issuing the next query.remember : if you want to query 5 different exercises, you must do them one at a time, waiting for the system to return the data before issuing the next query.

ASK_USER
→ when an important piece of information is unavailable from the database.

FINALIZE_DIAGNOSIS
→ when enough evidence has been gathered to make a reasonable conclusion,
  including the possibility that no program change is necessary.

Do not ask multiple redundant questions.

Do not repeatedly investigate a hypothesis that has already been reasonably
ruled out.

--------------------------------------------------

DATABASE DIRECTORY

When using QUERY_DATABASE, you MUST select the exact exercise name from:

{valid_exercises}

--------------------------------------------------

COACHING DIRECTIVE

If a user reports joint pain or secondary muscle fatigue (for example,
lower-back fatigue during squats), investigate relevant related heavy
compound lifts with QUERY_DATABASE before asking the user for information
that could already be obtained from the database.

However, this directive does not override the general principle of choosing
the most informative next action.

--------------------------------------------------

FINALIZATION RULES

When using FINALIZE_DIAGNOSIS:

- State the most supported explanation.
- Clearly distinguish evidence from uncertainty.
- State whether the issue appears temporary or persistent.
- State whether the normal progression target should remain unchanged.
- If recommending a temporary intervention, clearly identify it as temporary.
- Do not change the program merely because performance was poor once.
- Do not force a diagnosis when evidence is insufficient.
- Generate a detailed entry in the `issue_log_updates` array for EVERY
  exercise that was actually adjusted.
- If no exercise was adjusted, the issue log should reflect that no program
  change was made rather than inventing an adjustment.

--------------------------------------------------

OUTPUT FORMAT

You MUST respond in strict JSON format matching this exact schema:

{schema_instructions}

CRITICAL FORMATTING RULE: 
    You must output EXACTLY ONE JSON object per turn. Never output multiple JSON objects back-to-back. If you need to use multiple tools (e.g., querying two different exercises) or querying some exercises and current workout plan / previous plan, you MUST do them sequentially: output ONE tool call, wait for the system to reply with the data, and then output the next tool call in your next turn.
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
                temperature=0.0,
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
            elif ai_action.action_type == "QUERY_PLAN":
                print(f"\n📋 AI is looking up plan details: {ai_action.query_type.upper()}")
                print(f"   Reasoning: {ai_action.reasoning}")
                
                plan_result = fetch_plan_history(ai_action.query_type)
                
                messages.append({
                    "role": "user", 
                    "content": f"SYSTEM DATABASE RESULT (Plan Details - {ai_action.query_type}): {json.dumps(plan_result)}"
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
        run_pipeline('2026-01-27', api_key)