import pandas as pd
import json
import os
import time
import csv
from google import genai
from google.genai import types
# Import your cleanly separated logic
from engine import ProgressionEngine
from pydantic import TypeAdapter
from schema import AgentResponse  # Schema without InvestigationState
# ==========================================
# CUSTOM JSON ENCODER
# ==========================================
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)
# ==========================================
# DATABASE HELPER FUNCTIONS
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
        return {
            "active_issues": [],
            "past_rectified_issues": []
        }
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
    session = logs[
        (logs['date'] == target_date) &
        (logs['muscle_group'] == muscle_group)
    ].copy()
    if session.empty:
        return []
    session['set_index'] = session.groupby('exercise').cumcount() + 1
    flagged_tuples = {
        (item['exercise'], item['set_index'])
        for item in flagged_queue_items
    }
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
def fetch_exercise_history(
    exercise_name: str,
    filepath='workout_logs.csv',
    session_limit=3
):
    """Fetches all set logs for an exercise across the last N distinct workout sessions."""
    try:
        logs = pd.read_csv(filepath)
        ex_logs = logs[logs['exercise'] == exercise_name]
        if ex_logs.empty:
            return f"No recent data found for {exercise_name}."
        # Get unique dates sorted chronologically,
        # then take the last N sessions.
        unique_dates = sorted(
            ex_logs['date'].dropna().unique()
        )
        last_dates = unique_dates[-session_limit:]
        recent_logs = ex_logs[
            ex_logs['date'].isin(last_dates)
        ]
        return recent_logs.to_dict(orient='records')
    except Exception as e:
        return f"Database Error: {e}"
def fetch_plan_history(query_type: str, filepath='workout_plans.csv'):
    """Fetches the current active plan or the most recent previous plan."""
    try:
        plans = pd.read_csv(filepath)
        plan_cols = [
            'order',
            'exercise',
            'lower_reps',
            'upper_reps',
            'rir',
            'weight_increment(kg)',
            'current_target_weight(kg)',
            'current_target_reps'
        ]
        if query_type == "current":
            current_plan = plans[
                plans['end_date'].isna()
            ]
            if current_plan.empty:
                return "No active plan found."
            structured_plan = (
                current_plan
                .groupby('day_name')
                .apply(
                    lambda x: x.sort_values('order')[plan_cols].to_dict('records')
                )
                .to_dict()
            )
            return {
                "status": "current",
                "start_date": current_plan['start_date'].iloc[0],
                "schedule": structured_plan
            }
        elif query_type == "previous":
            past_plans = plans[
                plans['end_date'].notna()
            ]
            if past_plans.empty:
                return "No previous plans found."
            last_plan_id = (
                past_plans
                .sort_values('end_date', ascending=False)
                .iloc[0]['plan_id']
            )
            last_plan_data = past_plans[
                past_plans['plan_id'] == last_plan_id
            ]
            structured_plan = (
                last_plan_data
                .groupby('day_name')
                .apply(
                    lambda x: x.sort_values('order')[plan_cols].to_dict('records')
                )
                .to_dict()
            )
            return {
                "status": "previous",
                "plan_id": last_plan_id,
                "start_date": last_plan_data['start_date'].iloc[0],
                "end_date": last_plan_data['end_date'].iloc[0],
                "schedule": structured_plan
            }
    except Exception as e:
        return f"Database Error: {e}"
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
        "sleep": "unknown",
        "fatigue": "unknown",
        "physical_activity": "unknown",
        "nutrition": "unknown",
        "stress": "unknown",
        "soreness_pain": "unknown",
        "other_notes": "No data recorded for this session."
    }
# ==========================================
# 3. THE AGENTIC LOOP
# ==========================================
def call_ai_coach(initial_user_message, ai_payload, client):
    print("\n[INITIALIZING GEMINI AGENT (CHAT METHOD)...]")
    agent_adapter = TypeAdapter(AgentResponse)
    schema_instructions = json.dumps(agent_adapter.json_schema(), indent=2)
    valid_exercises = get_user_exercise_catalog()
    system_prompt = f"""
You are an expert strength coach investigating a user's stalled progress,
performance decline, or reported training problem.

Your goal is to investigate the root cause of the problem and prescribe
target adjustments, lifestyle adjustments, or maintain current targets,
the way a rational coach would.

--------------------------------------------------
AVAILABLE COMMANDS
--------------------------------------------------

1. "QUERY_DATABASE"

Look up recent performance history for a specific exercise.

Use this when historical performance of another relevant exercise could help:
- distinguish between competing explanations,
- determine whether the problem is isolated or affects related exercises,
- evaluate a cross-day carryover effect,
- or provide evidence that is missing from the current session.

Do NOT query an exercise if its relevant history is already provided in
pre_fetched_exercise_history.

You may query multiple independent exercises in one turn.

Do not wait for the user to mention another exercise before querying it if
that exercise could materially help test the current hypothesis.

2. "ASK_USER"

Ask the user for highly specific information that is not reliably available
in the database.

You MUST ask the user when:
- the likely cause is not strongly supported by the available evidence,
- multiple plausible causes remain,
- pain, injury, discomfort, or a subjective factor may be contributing,
- a previous issue may have recurred,
- or the proposed prescription depends on information that needs confirmation.

You MAY finalize without asking when:
- the issue is a minor performance fluctuation,
- the explanation is strongly supported by the available evidence,
- and no consequential intervention is being made.

Ask the single question with the highest value for distinguishing the
remaining plausible explanations.

3. "QUERY_PLAN"

Look up the user's previous workout plan when a program change, exercise
order, volume, frequency, or exercise selection could plausibly explain
the problem.

The current plan is already provided, so do not query the current plan.

4. "FINALIZE_DIAGNOSIS"

Finish the investigation and prescribe target adjustments, lifestyle
adjustments, or maintain current targets.

Do not finalize while important evidence is still missing.

--------------------------------------------------
INVESTIGATION PRINCIPLES
--------------------------------------------------

1. RETRIEVE BEFORE ASKING

First use the information already provided.

Then identify what important information is still missing.

Use QUERY_DATABASE or QUERY_PLAN when historical information could materially
help validate or reject a hypothesis.

Use ASK_USER when the missing information is something the database cannot
reliably provide.

2. DO NOT FORCE A DIAGNOSIS

If evidence is insufficient, continue investigating.

Distinguish between:
- strongly supported explanation
- plausible explanation
- unresolved possibility

If a cause is only a hypothesis, treat it as a hypothesis and verify it
before making a consequential intervention.

3. INVESTIGATE PATTERNS

Determine whether the problem is:
- isolated to one exercise,
- affecting related exercises,
- affecting several exercises across the same muscle group,
- or potentially caused by something from another training day.

Historical data from another exercise or another training day may be relevant
even when that exercise is not part of today's session.

4. CONSIDER TEMPORARY VS PERSISTENT PROBLEMS

A single bad session does not automatically justify changing targets.
Distinguish acute fluctuations from persistent trends. change plans only if the problem is persistent or severe enough to justify a consequential intervention.

5. CONFIRM BEFORE CONSEQUENTIAL INTERVENTIONS

Before changing a user's target based mainly on a hypothesis, verify the
hypothesis . 

If the proposed adjustment depends on a hypothesis that is not strongly
supported:
- ask the user a targeted question to confirm or refute it,
- retrieve any additional relevant exercise history,
- retrieve the previous plan if a program change could be involved,
- then reassess before prescribing.
even if it is decently supported by the available evidence, it is better to be safe and ask the user / check previous plan / exercise history of other exercises to be sure .

Do not change targets merely because a plausible explanation exists.

6. PLAN SET / ORDER HANDLING

In the workout plan:
- order = global execution order
- set_index = local set number for that exercise

Different sets of the same exercise may have different targets, weights,
reps, and RIR.

Always reference the correct set_index when making adjustments.

--------------------------------------------------
PARALLEL INVESTIGATION
--------------------------------------------------

When several independent pieces of information could help resolve the case,
perform them in the SAME turn.

For example, if a performance decline could be explained by:
- a related exercise's historical trend,
- a recent program change,
- and an unrecorded user-specific event,

you may output all three actions together:

- QUERY_DATABASE
- QUERY_PLAN
- ASK_USER

These actions are independent when none requires the result of another.

Do NOT wait for one result before requesting another independent result.
sometimes , if you need only one of them ( like QUERY_DATABASE or QUERY_PLAN  ) it is still better to ask user aswell in the same turn to be sure about the hypothesis and to avoid any misdiagnosis.
After the results are returned, reassess the complete evidence and then
FINALIZE_DIAGNOSIS in a later turn.

Do not finalize in the same turn as an ASK_USER or a database/plan query when
those results are necessary to resolve the hypothesis.

--------------------------------------------------
DATABASE DIRECTORY
--------------------------------------------------

When using QUERY_DATABASE, use the exact exercise name from:

{valid_exercises}
--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Respond in strict JSON matching this schema:

{schema_instructions}

--------------------------------------------------
MULTI-QUERY BATCHING
--------------------------------------------------

You may include multiple independent actions in "actions".

A single turn may contain:
- multiple QUERY_DATABASE actions,
- QUERY_DATABASE + QUERY_PLAN,
- QUERY_DATABASE + ASK_USER,
- QUERY_PLAN + ASK_USER,
- or QUERY_DATABASE + QUERY_PLAN + ASK_USER.

Only batch actions when they are independent.

Do not batch actions when one action depends on the result of another.

When the investigation requires these results, gather them first and
finalize only after the next turn has access to the returned information.
"""
    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            response_mime_type="application/json",
        )
    )
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    interaction_count = 0
    total_cached_tokens = 0
    pending_message = initial_user_message
    while True:
        try:
            time.sleep(2)
            response = chat.send_message(pending_message)
            usage = response.usage_metadata
            print(f"usage: {usage}")
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens_turn = getattr(usage, "total_token_count", 0) or 0
            cached_this_turn = getattr(usage, "cached_content_token_count", 0) or 0
            total_tokens += total_tokens_turn
            total_input_tokens += prompt_tokens
            total_output_tokens += output_tokens
            total_cached_tokens += cached_this_turn
            interaction_count += 1
            print(f"📊 Tokens: {prompt_tokens} In | {output_tokens} Out | {total_tokens_turn} Total | Cached this turn: {cached_this_turn} | Cached cumulative: {total_cached_tokens}")
            response_content = response.text
            print("\n🔎 RAW GEMINI RESPONSE:")
            print(response_content)
            print()
            parsed_envelope = agent_adapter.validate_json(response_content)
            actions_list = parsed_envelope.actions
            system_results = []
            should_continue = False
            for action in actions_list:
                if action.action_type == "QUERY_DATABASE":
                    print(f"🔍 AI is looking up history for: {action.exercise_to_query}")
                    data = fetch_exercise_history(action.exercise_to_query)
                    system_results.append(f"Database ({action.exercise_to_query}): {json.dumps(data, cls=CustomJSONEncoder)}")
                    should_continue = True
                elif action.action_type == "QUERY_PLAN":
                    print(f"📋 AI is looking up plan details: {action.query_type.upper()}")
                    data = fetch_plan_history(action.query_type)
                    system_results.append(f"Plan ({action.query_type}): {json.dumps(data, cls=CustomJSONEncoder)}")
                    should_continue = True
                elif action.action_type == "ASK_USER":
                    print(f"🧠 AI Reasoning: {action.rationale}")
                    print(f"🗣️ COACH: {action.question}")
                    user_answer = input("👉 YOUR ANSWER: ")
                    system_results.append(f"USER ANSWER: {user_answer}")
                    should_continue = True
                    break
                elif action.action_type == "FINALIZE_DIAGNOSIS":
                    print("total tokens used in this session: ", total_tokens)
                    print(f"Total Input Tokens: {total_input_tokens}")
                    print(f"Total Output Tokens: {total_output_tokens}")
                    print(f"Total Cached Input Tokens: {total_cached_tokens}")
                    print(f"Number of User Interactions: {interaction_count}")
                    return action.model_dump()
            if should_continue and system_results:
                combined_results = "\n---\n".join(system_results)
                pending_message = "SYSTEM RESULTS / USER UPDATE:\n" + combined_results
                print("📜 [History Appended to Chat]")
            else:
                return None
        except Exception as e:
            print(f"Agent Loop Error: {e}")
            return None
# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def run_pipeline(target_date, api_key):
    client = genai.Client(
        api_key=api_key
    )
    engine = ProgressionEngine()
    print(
        f"\n--- RUNNING SESSION ANALYSIS FOR "
        f"{target_date} ---"
    )
    fast_path_updates, ai_routing_queue = (
        engine.sweep_session(target_date)
    )
    if ai_routing_queue:
        print(
            "\n⚠️ AI INTERVENTION REQUIRED:"
        )
        grouped_issues = {}
        for item in ai_routing_queue:
            mg = item['muscle_group']
            if mg not in grouped_issues:
                grouped_issues[mg] = []
            grouped_issues[mg].append(item)
        daily_context = fetch_session_context(
            target_date
        )
        for mg, flags in grouped_issues.items():
            print(
                f"\nGathering context for muscle group: "
                f"[{mg}]..."
            )
            history = get_issue_history(
                mg
            )
            # ==========================================
            # PREFETCH FLAGGED EXERCISE HISTORY
            # ==========================================
            pre_fetched_histories = {}
            flagged_exercise_names = []
            for flag in flags:
                ex_name = flag['exercise']
                if ex_name not in flagged_exercise_names:
                    flagged_exercise_names.append(
                        ex_name
                    )
                    pre_fetched_histories[
                        ex_name
                    ] = fetch_exercise_history(
                        ex_name,
                        session_limit=3
                    )
            # ==========================================
            # PREFETCH CURRENT PLAN
            # ==========================================
            current_plan_data = (
                fetch_plan_history(
                    "current"
                )
            )
            # ==========================================
            # BUILD PAYLOAD
            # ==========================================
            ai_payload = {
                "muscle_group": mg,
                "flagged_sets": flags,
                "healthy_sets_today":
                    get_healthy_sets(
                        target_date,
                        mg,
                        'workout_logs.csv',
                        flags
                    ),
                "pending_fast_path_updates":
                    fast_path_updates,
                "global_session_context":
                    daily_context,
                "pre_fetched_current_plan":
                    current_plan_data,
                "pre_fetched_exercise_history":
                    pre_fetched_histories,
                "active_issues":
                    history["active_issues"],
                "past_rectified_issues":
                    history["past_rectified_issues"]
            }
            # ==========================================
            # INITIAL USER MESSAGE
            # ==========================================
            initial_user_message = f"""
<session_context date="{target_date}" muscle_group="{mg}">
<data_inventory_manifest>

Flagged exercise histories are already loaded for:
{chr(10).join(
    f"- {name}"
    for name in flagged_exercise_names
)}

Do NOT query those flagged exercises again.

You MAY use QUERY_DATABASE for other relevant exercises when their history
could help determine whether the problem is isolated or part of a broader
pattern across related exercises.

You MAY use QUERY_PLAN for the previous plan when the start of current plan is recent enough / when a program change could plausibly explain the problem.
The current plan is already provided.
</data_inventory_manifest>

<global_session_context>
{json.dumps(
    daily_context,
    cls=CustomJSONEncoder,
    indent=2
)}
</global_session_context>

<flagged_sets>
{json.dumps(
    flags,
    cls=CustomJSONEncoder,
    indent=2
)}
</flagged_sets>

<healthy_sets_today>
{json.dumps(
    ai_payload["healthy_sets_today"],
    cls=CustomJSONEncoder,
    indent=2
)}
</healthy_sets_today>

<pre_fetched_current_plan>
{json.dumps(
    current_plan_data,
    cls=CustomJSONEncoder,
    indent=2
)}
</pre_fetched_current_plan>

<pre_fetched_exercise_history>
{json.dumps(
    pre_fetched_histories,
    cls=CustomJSONEncoder,
    indent=2
)}
</pre_fetched_exercise_history>

<active_issues>
{json.dumps(
    history["active_issues"],
    cls=CustomJSONEncoder,
    indent=2
)}
</active_issues>

<past_rectified_issues>
{json.dumps(
    history["past_rectified_issues"],
    cls=CustomJSONEncoder,
    indent=2
)}
</past_rectified_issues>

<pending_fast_path_updates>
{json.dumps(
    fast_path_updates,
    cls=CustomJSONEncoder,
    indent=2
)}
</pending_fast_path_updates>

</session_context>
"""
            print(f"initial user message: {initial_user_message}")
            ai_prescription = call_ai_coach(
                initial_user_message,
                ai_payload,
                client
            )
            # ==========================================
            # PRINT FINAL PRESCRIPTION
            # ==========================================
            if ai_prescription:
                print(
                    "\n✅ FINAL AI PRESCRIPTION:"
                )
                print(
                    f"  Diagnosis: "
                    f"{ai_prescription['analysis']}"
                )
                for adj in ai_prescription[
                    'adjustments'
                ]:
                    set_idx = adj.get(
                        'set_index',
                        'N/A'
                    )
                    print(
                        f"  - "
                        f"{adj['exercise']} "
                        f"(Set {set_idx}) "
                        f"Target Updated: "
                        f"{adj['new_target_weight_kg']}"
                        f"kg x "
                        f"{adj['new_target_reps']} "
                        f"@ RIR "
                        f"{adj['new_target_rir']}"
                    )
                    print(
                        f"  - Cue: "
                        f"{adj['ai_instructions']}"
                    )
                print(
                    "\n📝 PROPOSED ISSUE LOG UPDATES:"
                )
                for log_update in ai_prescription.get(
                    'issue_log_updates',
                    []
                ):
                    generated_advice = (
                        log_update.get(
                            'ai_advice',
                            'No advice recorded.'
                        )
                    )
                    print(
                        f'  - APPEND TO issue_log.csv: '
                        f'{target_date} | '
                        f'{log_update.get("exercise")} | '
                        f'{mg} | '
                        f'{log_update.get("issue_type", "Problem")} | '
                        f'{log_update.get("description", "")} | '
                        f'Active | '
                        f'"{generated_advice}"'
                    )
    else:
        if fast_path_updates:
            print(
                "\n✅ FAST PATH "
                "(DETERMINISTIC INCREMENTS):"
            )
            for update in fast_path_updates:
                print(
                    f"  - "
                    f"{update['exercise']} "
                    f"(Set {update['order']}): "
                    f"target updated to "
                    f"{update['new_target_weight_kg']}kg x "
                    f"{update['new_target_reps']}"
                )
# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )
    if not api_key:
        print(
            "Error: GEMINI_API_KEY environment variable "
            "not found."
        )
        print(
            'Run this in your terminal first: '
            '$env:GEMINI_API_KEY="your_key_here"'
        )
    else:
        run_pipeline(
            '2026-02-01',
            api_key
        )
#very poor,very high,high,poor,high,high,Poor recovery and noticeable soreness before training.
